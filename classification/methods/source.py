import torch
from copy import deepcopy
from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY
import torch.cuda as cuda
from flops_profiler.profiler import get_model_profile
import time
from flops_profiler.profiler import FlopsProfiler
@ADAPTATION_REGISTRY.register()
class Source(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

    @torch.no_grad()
    def forward_and_adapt(self, x):
        # prof = FlopsProfiler(self.model, None)
        # prof.start_profile()
        imgs_test = x[0]
        # prof.stop_profile()
        # flops = prof.get_total_flops()
        # macs = prof.get_total_macs()
        # params = prof.get_total_params()
        # prof.print_model_profile(profile_step=1)
        # prof.end_profile()
        return self.model(imgs_test)

    def copy_model_and_optimizer(self):
        """Copy the model and optimizer states for resetting after adaptation."""
        model_states = [deepcopy(model.state_dict()) for model in self.models]
        optimizer_state = None
        return model_states, optimizer_state

    def reset(self):
        for model, model_state in zip(self.models, self.model_states):
            model.load_state_dict(model_state, strict=True)

    def configure_model(self):
        self.model.eval()
        self.model.requires_grad_(False)
