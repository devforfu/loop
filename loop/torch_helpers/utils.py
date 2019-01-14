"""
A set of tools to operate with torch modules.
"""
import sys
from enum import Enum

import torch
from torch import nn

from ..utils import classname


class LayerState(Enum):
    """Training status of torch layer."""

    NoParams = 0
    Trainable = 1
    Frozen = 2

    def __str__(self):
        return {0: '-', 1: 'ok', 2: 'frozen'}[self.value]


def flat_model(model: nn.Module) -> list:
    """Converts PyTorch model from hierarchical representation into a single
    list of modules.
    """
    def flatten(m):
        children = list(m.children())
        if not children:
            return [m]
        return sum([flatten(child) for child in children], [])

    return flatten(model)


def as_sequential(model: nn.Module) -> nn.Module:
    """Converts model with nested submodules into Sequential model."""

    return nn.Sequential(*list(model.children()))


def get_output_shape(model: nn.Module) -> list:
    """Passes a dummy input tensor through the sequential model to get the
    shape of its output tensor (batch dimension is not included).
    """
    first, *rest = flat_model(model)
    shape = first.in_channels, 128, 128
    dummy_input = torch.zeros(shape)
    out = model(dummy_input[None])
    return list(out.size())[1:]


def training_status(layer: nn.Module) -> LayerState:
    """Returns module's training status."""

    params = list(layer.parameters())
    if not params:
        return LayerState.NoParams
    trainable = sum([1 for p in params if p.requires_grad]) > 0
    return LayerState.Trainable if trainable else LayerState.Frozen


def freeze_status(m: nn.Module, stream=sys.stdout):
    """Flattens the model and prints its require_grad value."""

    stream.write('Layers status\n')
    stream.write('-' * 80 + '\n')
    for layer in flat_model(m):
        name = layer.__class__.__name__
        status = training_status(layer)
        stream.write(f'{name:<71} [{status:6s}]\n')
    stream.flush()


def unfreeze_layers(m: nn.Module):
    """Unfreezes all trainable layers in the model."""

    requires_grad(m, True)


def freeze_layers(m: nn.Module):
    """Freezes all trainable layers in the model."""

    requires_grad(m, False)


def requires_grad(m: nn.Module, grad: bool=True):
    for layer in flat_model(m):
        status = training_status(layer)
        if status in (LayerState.Trainable, LayerState.Frozen):
            layer.requires_grad = grad


def classifier_weights(m: nn.Module, bn=(1, 1e-3)):
    """Initializes layers weights for a classification model."""

    name = classname(m)

    with torch.no_grad():
        if name.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)

        elif name.find('BatchNorm') != -1:
            weight, bias = bn
            nn.init.constant_(m.weight, weight)
            nn.init.constant_(m.bias, bias)

        elif name == 'Linear':
            nn.init.kaiming_normal_(m.weight)
            nn.init.zeros_(m.bias)


def get_activation_layer(activ, inplace=True):
    if activ == 'relu':
        return nn.ReLU(inplace)
    elif activ == 'leaky_relu':
        return nn.LeakyReLU(inplace=inplace)
    return activ
