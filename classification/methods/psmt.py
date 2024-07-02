import os
import torch
import torch.nn as nn
import torch.jit
from methods.base import TTAMethod
from augmentations.transforms_cotta import get_tta_transforms
from utils.registry import ADAPTATION_REGISTRY
from datasets.data_loading import get_source_loader
import logging
from models.model import split_up_model
import torch.cuda as cuda
from flops_profiler.profiler import get_model_profile

logger = logging.getLogger(__name__)


class SymmetricCrossEntropy(nn.Module):
    def __init__(self, alpha=0.5):
        super(SymmetricCrossEntropy, self).__init__()
        self.alpha = alpha

    def __call__(self, x, x_ema):
        return -(1 - self.alpha) * (x_ema.softmax(1) * x.log_softmax(1)).sum(1) - self.alpha * (
                    x.softmax(1) * x_ema.log_softmax(1)).sum(1)


def find_quantile(arr, perc):
    arr_sorted = torch.sort(arr).values
    frac_idx = perc * (len(arr_sorted) - 1)
    frac_part = frac_idx - int(frac_idx)
    low_idx = int(frac_idx)
    high_idx = low_idx + 1
    quant = arr_sorted[low_idx] + (arr_sorted[high_idx] - arr_sorted[low_idx]) * frac_part  # linear interpolation

    return quant


def print_output(flops, macs, params):
    print('{:<30}  {:<8}'.format('Number of flops: ', flops))
    print('{:<30}  {:<8}'.format('Number of MACs: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))


def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


def update_ema_variables_two(ema_model, model, alpha_teacher, Mask):
    for ema_param, param, mask in zip(ema_model.parameters(), model.parameters(), Mask):
        ema_param.data = mask * param.data + (1 - mask) * (
                    alpha_teacher * ema_param.data + (1 - alpha_teacher) * param.data)
    return ema_model


@ADAPTATION_REGISTRY.register()
class PSMT(TTAMethod):
    def __init__(self, cfg, model, num_classes, perc=0.03):
        super().__init__(cfg, model, num_classes)
        print(cfg.CORRUPTION.DATASET)
        lamb = 500
        self.fisher_alpha = lamb
        self.mt = cfg.M_TEACHER.MOMENTUM
        self.rst = cfg.COTTA.RST
        self.ap = cfg.COTTA.AP
        self.n_augmentations = cfg.TEST.N_AUGMENTATIONS
        self.final_lr = cfg.OPTIM.LR
        # Setup EMA and anchor/source model
        self.model_ema = self.copy_model(self.model)
        for param in self.model_ema.parameters():
            param.detach_()
        self.model_anchor = self.copy_model(self.model)
        for param in self.model_anchor.parameters():
            param.detach_()
        arch_name = cfg.MODEL.ARCH
        self.symmetric_cross_entropy = SymmetricCrossEntropy()
        _, self.src_loader = get_source_loader(dataset_name=cfg.CORRUPTION.DATASET,
                                               adaptation=cfg.MODEL.ADAPTATION,
                                               preprocess=model.model_preprocess,
                                               data_root_dir=cfg.DATA_DIR,
                                               batch_size=200,
                                               ckpt_path=cfg.MODEL.CKPT_PATH,
                                               num_samples=cfg.SOURCE.NUM_SAMPLES,
                                               percentage=cfg.SOURCE.PERCENTAGE,
                                               workers=min(cfg.SOURCE.NUM_WORKERS, os.cpu_count()))
        self.src_loader_iter = iter(self.src_loader)
        self.models = [self.model, self.model_ema, self.model_anchor]
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()
        self.feature_extractor, self.classifier = split_up_model(self.model, arch_name, self.dataset_name)
        self.softmax_entropy = softmax_entropy_cifar if "cifar" in self.dataset_name else softmax_entropy_imagenet
        self.transform = get_tta_transforms(self.dataset_name)

        self.perc = perc
        self.flag = 0
        self.cnt = 0
        self.fisher2 = {}
        self.res = {}
        _, fisher_loader = get_source_loader(dataset_name=cfg.CORRUPTION.DATASET,
                                             adaptation=cfg.MODEL.ADAPTATION,
                                             preprocess=model.model_preprocess,
                                             data_root_dir=cfg.DATA_DIR,
                                             batch_size=cfg.TEST.BATCH_SIZE,
                                             ckpt_path=cfg.MODEL.CKPT_PATH,
                                             num_samples=10000,  # number of samples for ewc reg.
                                             percentage=cfg.SOURCE.PERCENTAGE,
                                             workers=min(cfg.SOURCE.NUM_WORKERS, os.cpu_count()))
        ewc_optimizer = torch.optim.SGD(self.params, 0.01)
        self.fishers = {}
        train_loss_fn = nn.CrossEntropyLoss().to(self.device)
        for iter_, batch in enumerate(fisher_loader, start=1):
            images = batch[0].to(self.device, non_blocking=True)
            outputs = self.model(images)
            _, targets = outputs.max(1)
            loss = train_loss_fn(outputs, targets)
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if iter_ > 1:
                        fisher = param.grad.data.clone().detach() ** 2 + self.fishers[name][0]
                    else:
                        fisher = param.grad.data.clone().detach() ** 2
                    if iter_ == len(fisher_loader):
                        fisher = fisher / iter_
                    self.fishers.update({name: [fisher, param.data.clone().detach()]})
            ewc_optimizer.zero_grad()
        logger.info("Finished computing the fisher matrices...")
        del ewc_optimizer


    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        imgs_test = x[0]
        self.optimizer.zero_grad()
        student_preds = self.model(imgs_test)
        teacher_preds = self.model_ema(imgs_test)

        # forward original test data
        features_test = self.feature_extractor(imgs_test)
        outputs_test = self.classifier(features_test)
        # forward augmented test data
        features_aug_test = self.feature_extractor(self.transform((imgs_test)))
        features_aug_test = self.feature_extractor(imgs_test)
        outputs_aug_test = self.classifier(features_aug_test)
        # Create the prediction of the anchor (source) model
        anchor_prob = torch.nn.functional.softmax(self.model_anchor(imgs_test), dim=1).max(1)[0]

        # Augmentation-averaged Prediction
        outputs_emas = []
        if anchor_prob.mean(0) < self.ap:
            for _ in range(self.n_augmentations):
                outputs_ = self.model_ema(imgs_test).detach()
                # outputs_ = self.model_ema(self.transform(imgs_test)).detach()
                outputs_emas.append(outputs_)

            # Threshold choice discussed in supplementary
            outputs_ema = torch.stack(outputs_emas).mean(0)
        else:
            # Create the prediction of the teacher model
            outputs_ema = self.model_ema(imgs_test)
        # Student update
        loss = (0.5 * self.symmetric_cross_entropy(outputs_test, outputs_ema) + 0.5 * self.symmetric_cross_entropy(
            outputs_aug_test, outputs_ema)).mean(0)
        if self.fishers is not None:
            ewc_loss = 0
            for name, param in self.model.named_parameters():
                if name in self.fishers:
                    ewc_loss += self.fisher_alpha * (self.fishers[name][0] * (param - self.fishers[name][1]) ** 2).sum()
            loss += ewc_loss
        loss.backward()

        fisher_dict = {}
        for nm, m in self.model.named_modules():  ## previously used model, but now using self.model
            for npp, p in m.named_parameters():
                if npp in ['weight', 'bias'] and p.requires_grad:
                    fisher_dict[f"{nm}.{npp}"] = p.grad.data.clone().pow(2)
        fisher_list = []
        for name in fisher_dict:
            fisher_list.append(fisher_dict[name].reshape(-1))
        fisher_flat = torch.cat(fisher_list)
        threshold = find_quantile(fisher_flat, self.perc)

        self.optimizer.step()
        self.optimizer.zero_grad()

        # Teacher update
        self.model_ema = update_ema_variables(ema_model=self.model_ema, model=self.model, alpha_teacher=self.mt)

        # Stochastic restore
        if self.rst > 0.:
            MASK = []
            for nm, m in self.model.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        # fisher_info = str(fisher_dict[f"{nm}.{npp}"].cpu().numpy())
                        mask_fish = (fisher_dict[
                                         f"{nm}.{npp}"] < threshold).float().cuda()  # masking makes it restore candidate
                        mask = mask_fish
                        MASK.append(mask)
                        with torch.no_grad():
                            p.data = self.model_states[0][f"{nm}.{npp}"] * mask + p * (1. - mask)
        self.model_ema = update_ema_variables_two(ema_model=self.model_ema, model=self.model, alpha_teacher=self.mt,
                                                  Mask=MASK)
        return outputs_ema + outputs_test

    @torch.no_grad()
    def forward_sliding_window(self, x):
        """
        Create the prediction for single sample test-time adaptation with a sliding window
        :param x: The buffered data created with a sliding window
        :return: Model predictions
        """
        imgs_test = x[0]
        return self.model_ema(imgs_test)

    def configure_model(self):
        """Configure model."""
        # self.model.train()
        self.model.eval()  # eval mode to avoid stochastic depth in swin. test-time normalization is still applied
        # disable grad, to (re-)enable only what we update
        self.model.requires_grad_(False)
        # enable all trainable
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()  # always forcing train mode in bn1d will cause problems for single sample tta
                m.requires_grad_(True)
            else:
                m.requires_grad_(True)


@torch.jit.script
def softmax_entropy_cifar(x, x_ema) -> torch.Tensor:
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)


@torch.jit.script
def softmax_entropy_imagenet(x, x_ema) -> torch.Tensor:
    return -0.5 * (x_ema.softmax(1) * x.log_softmax(1)).sum(1) - 0.5 * (x.softmax(1) * x_ema.log_softmax(1)).sum(1)
