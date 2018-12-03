import torch
from torch import nn
from torchvision import models

from .model import as_sequential, classifier_weights, get_activation_layer
from .model import classname, get_output_shape


class AdaptiveConcatPool2d(nn.Module):
    """Applies average and maximal adaptive pooling to the tensor and
    concatenates results into a single tensor.

    The idea is taken from fastai library.
    """
    def __init__(self, size=1):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(size)
        self.max = nn.AdaptiveMaxPool2d(size)

    def forward(self, x):
        return torch.cat([self.max(x), self.avg(x)], 1)


class Flatten(nn.Module):
    """Converts N-dimensional tensor into 'flat' one."""

    def forward(self, x):
        return x.view(x.size(0), -1)


class LinearGroup(nn.Module):
    """The linear layer with 'batteries included'.

    The layer creates not only the linear layer but:
        * (optionally) BatchNorm
        * (optionally) Dropout
        * Linear
        * Activation (ReLU, LeakyReLU, None, or custom)

    """
    def __init__(self, ni, no, dropout=None, bn=True, activ='relu'):
        super().__init__()
        layers = [nn.BatchNorm1d(ni)] if bn else []
        layers.append(nn.Linear(ni, no))
        if dropout is not None and dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(get_activation_layer(activ))
        self.main = nn.Sequential(layers)

    def forward(self, x):
        return self.main(x)


class Classifier(nn.Module):
    """Builds a simple classifier based on pretrained architecture."""

    def __init__(self, n_classes, top=None, arch=models.resnet34,
                 init_fn=classifier_weights):

        super().__init__()

        def create_top(conf):
            for params in conf:
                ni, no = params['ni'], params['no']
                drop, bn = params.get('drop', 0.25), params.get('bn', True)
                yield LinearGroup(ni, no, drop, bn, 'leaky_relu')
            yield nn.Linear(conf[-1]['no'], n_classes)


        model = arch(True)
        seq_model = as_sequential(model)
        backbone, classifier = seq_model[:-2], seq_model[-2:]
        out_shape = get_output_shape(backbone)
        input_size = out_shape[0] * 2

        if top is None:
            top = [
                {'ni': input_size, 'no': 512, 'drop': 0.25},
                {'ni': 512, 'no': 256, 'drop': 0.5}
            ]

        self.backbone = backbone
        self.bottleneck = nn.Sequential(AdaptiveConcatPool2d(), Flatten())
        self.top = nn.Sequential(*list(create_top(top)))
        self.init(init_fn)

    def forward(self, x):
        return self.top(self.bottleneck(self.backbone(x)))

    def init(self, func):
        if func is not None:
            self.apply(func)

    def freeze_backbone(self, freeze=True, bn=True):
        for child in self.backbone.children():
            name = classname(child)
            if not bn and name.find('BatchNorm') != -1:
                continue
            for p in child.parameters():
                p.requires_grad = not freeze
