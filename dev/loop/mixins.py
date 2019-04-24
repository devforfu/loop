# -----------------------------------------
# THIS FILE WAS AUTOGENERATED! DO NOT EDIT!
# -----------------------------------------
# file to edit: 00d_mixins.ipynb

from inspect import signature

from loop.utils import classname


class ParametersMixin:
    """Adds to a class methods to set and get init keywords parameters.

    Almost entirely copied from sklearn setters/getters.
    """
    def set_params(self, **kwargs):
        available_params = set(self.get_params())
        cls_name = classname(self)
        for k, v in kwargs.items():
            if k not in available_params:
                raise ValueError(f'the property \'{k}\' is invalid for {cls_name} objects')
            setattr(self, k, v)

    def get_params(self) -> dict:
        if self.__class__.__init__ is object.__init__:
            return {}
        init_signature = signature(self.__init__)
        params = [p for p in init_signature.parameters.values()
                  if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        return {p.name: getattr(self, p.name) for p in params}

    def clean_params(self, params: dict) -> dict:
        """Removes keys from the dictionary that doesn't present in object's initializer
        signature.
        """
        keys = set(self.get_params())
        return {k: v for k, v in params.items() if k in keys}

    def __repr__(self):
        params = ', '.join([f'{k}={v}' for k, v in self.get_params().items()])
        return f'<{classname(self)}: {params}>'
