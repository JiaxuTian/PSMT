from typing import Any


class Registry:
    def __init__(self, name: str) -> None:
        self._name = name
        self._obj_map = dict()

    def _do_register(self, name: str, obj: Any) -> None:
        assert name not in self._obj_map, \
            "An object named '{}' was already registered in '{}' registry!".format(name, self._name)
        self._obj_map[name] = obj

    def register(self, obj: Any = None) -> Any:
        if obj is None:
            # used as a decorator
            def deco(func_or_class: Any) -> Any:
                name = func_or_class.__name__
                self._do_register(name, func_or_class)
                if name != name.lower():
                    self._do_register(name.lower(), func_or_class)
                return func_or_class

            return deco

        # used as a function call
        name = obj.__name__
        self._do_register(name, obj)
        if name != name.lower():
            self._do_register(name.lower(), obj)

    def get(self, name: str) -> Any:
        if name not in self._obj_map:
            raise KeyError('Object name "{}" does not exist in "{}" registry'.format(name, self._name))

        return self._obj_map[name]

    def registered_names(self):
        return list(self._obj_map.keys())


ADAPTATION_REGISTRY = Registry("ADAPTATION")
