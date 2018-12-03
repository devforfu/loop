import sys

import torch
from torch import nn


def flat_model(model: nn.Module):
    """Converts PyTorch model from hierarchical representation into a single
    list of modules.
    """
    def flatten(m):
        children = list(m.children())
        if not children:
            return [m]
        return sum([flatten(child) for child in children], [])

    return flatten(model)


def as_sequential(model: nn.Module):
    """Converts model with nested submodules into Sequential model."""

    return nn.Sequential(*list(model.children()))


def get_output_shape(model):
    """Passes a dummy input tensor through the sequential model to get the
    shape of its output tensor (batch dimension is not included).
    """
    first, *rest = flat_model(model)
    shape = first.in_channels, 128, 128
    dummy_input = torch.zeros(shape)
    out = model(dummy_input[None])
    return list(out.size())[1:]


def training_status(layer):
    params = list(layer.parameters())
    if not params:
        return 'no params'
    trainable = [p for p in params if p.requires_grad]
    return 'frozen' if not trainable else ''


def freeze_status(m, stream=sys.stdout):
    """Flattens the model and prints its require_grad value."""
    stream.write('Layers status\n')
    stream.write('-' * 80 + '\n')
    for layer in flat_model(m):
        name = layer.__class__.__name__
        status = training_status(layer)
        stream.write(f'{name:<68} [{status:9s}]\n')
    stream.flush()


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


def classname(x):
    return x.__class__.__name__
