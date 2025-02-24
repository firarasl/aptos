from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

from aptos.base import BaseModel
import timm


def cycle_channel_layers(weights, n):
    """Repeat channel weights n times. Assumes channels are dim 1."""
    slices = [(c % 3, c % 3 + 1) for c in range(n)]  # slice a:a+1 to keep dims
    new_weights = torch.cat([
        weights[:, a:b, :, :] for a, b in slices
    ], dim=1)
    return new_weights


class EffNet(BaseModel):
    """
    https://github.com/lukemelas/EfficientNet-PyTorch
    """
    def __init__(self, num_classes, pretrained, model='b2', verbose=0):
        super().__init__(verbose)
        model_name = f'efficientnet-{model}'
        if pretrained:
            self.model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
        else:
            self.model = EfficientNet.from_name(
                model_name,
                override_params={'num_classes': num_classes})

        # for name, w in self.model.named_parameters():
        #     if '_fc' not in name:
        #         w.requires_grad = False

        self.logger.info(f'<init>: \n{self}')

    def forward(self, x):
        return self.model(x)

    def __str__(self):
        return str(self.model)

    def __repr__(self):
        return self.__str__()


class EffNetMaxAvg(BaseModel):

    def __init__(self, num_classes, pretrained, model='b2', verbose=0):
        super().__init__(verbose)

        model_name = f'efficientnet-{model}'
        if pretrained:
            self.model = EfficientNetMaxAvg.from_pretrained(model_name, num_classes=num_classes)
        else:
            self.model = EfficientNetMaxAvg.from_name(
                model_name,
                override_params={'num_classes': num_classes})

        self.logger.info(f'<init>: \n{self}')

    def forward(self, x):
        return self.model(x)

    def __str__(self):
        return str(self.model)

    def __repr__(self):
        return self.__str__()


class EfficientNetMaxAvg(EfficientNet):
    """
    Modified EfficientNet to use concatenated Max + Avg pooling
    """

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__(blocks_args=blocks_args, global_params=global_params)

        fc = nn.Sequential(OrderedDict([
            ('bn1', nn.BatchNorm1d(self.model._bn1.num_features * 2)),
            ('drop1', nn.Dropout(p=self._dropout)),
            ('linear1', nn.Linear(self.model._bn1.num_features * 2, 512)),
            ('mish', Mish()),
            ('bn2', nn.BatchNorm1d(512)),
            ('drop2', nn.Dropout(p=self._dropout / 2)),
            ('linear2', nn.Linear(512, self._global_params.num_classes))
        ]))

        nn.init.kaiming_normal_(fc._modules['linear1'].weight)
        nn.init.kaiming_normal_(fc._modules['linear2'].weight)

        self._bn1 = AdaptiveMaxAvgPool()
        self._fc = fc

    def forward(self, x):
        x = self.extract_features(x)
        x = self._bn1(x)
        x = self._fc(x)
        return x


class AdaptiveMaxAvgPool(nn.Module):

    def __init__(self):
        super().__init__()
        self.ada_avgpool = nn.AdaptiveAvgPool2d(1)
        self.ada_maxpool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        avg_x = self.ada_avgpool(x)
        max_x = self.ada_maxpool(x)
        x = torch.cat((avg_x, max_x), dim=1)
        x = x.view(x.size(0), -1)
        return x


class Mish(nn.Module):
    """
    https://github.com/lessw2020/mish/blob/master/mish.py
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))


class EffNetVit(BaseModel):
    def __init__(self, model='efficientvit_mit_b2', num_classes=1, pretrained=True, verbose=0):
        super().__init__(verbose)
        model = 'efficientvit_b2.r256_in1k'
        if pretrained:
            self.model = timm.create_model(model, pretrained=True, num_classes=num_classes)
        else:
            self.model = timm.create_model(model, pretrained=False, num_classes=num_classes)

        self.logger.info(f'<init>: \n{self}')

    def forward(self, x):
        return self.model(x)

    def __str__(self):
        return str(self.model)

    def __repr__(self):
        return self.__str__()


