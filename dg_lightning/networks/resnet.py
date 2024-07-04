
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, resnet50, resnet101
from dg_lightning.networks.base import _ConvBackboneBase
from dg_lightning.networks.general import Flatten


class ResNetBackbone(_ConvBackboneBase):
    def __init__(self,
                 name: str = 'resnet50',
                 in_channels: int = 3,
                 pretrained: bool = False) -> None:
        super().__init__()

        self.name: str = name
        self.in_channels: int = in_channels
        self.pretrained: bool = pretrained

        _resnet = self._build_with_torchvision()
        self.layers = self.fetch_backbone_only(_resnet, gap_and_flatten=True)

        if self.in_channels != 3:
            """Change the input channels of the first convolution.
            TODO: In this case, restrict the use of pretrained models."""
            self.layers = self.change_first_conv_input_channels(self.layers, c=self.in_channels)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x)

    def _build_with_torchvision(self):
        # TODO: consider using torchvision.prototype.models
        # TODO: Reference: https://pytorch.org/blog/introducing-torchvision-new-multi-weight-support-api/
        if self.name == 'resnet18':
            return resnet18(pretrained=self.pretrained)
        elif self.name == 'resnet50':
            return resnet50(pretrained=self.pretrained)
        elif self.name == 'resnet101':
            return resnet101(pretrained=self.pretrained)
        else:
            raise NotImplementedError

    @staticmethod
    def fetch_backbone_only(resnet: nn.Module, gap_and_flatten: bool = True) -> nn.Module:
        model = nn.Sequential()
        for name, child in resnet.named_children():
            if name not in ['avgpool', 'fc']:
                model.add_module(name, child)
        
        if gap_and_flatten:
            model.add_module('gap', nn.AdaptiveAvgPool2d(1))
            model.add_module('flatten', Flatten())

        return model

    @staticmethod
    def remove_gap_and_fc(resnet: nn.Module) -> nn.Module:
        """
        Helper function which removes:
            1) global average pooling
            2) fully-connected head
        """
        model = nn.Sequential()
        for name, child in resnet.named_children():
            if name not in ['avgpool', 'fc']:
                model.add_module(name, child)

        return model

    @staticmethod
    def change_first_conv_input_channels(resnet: nn.Module, c: int) -> nn.Module:
        """Add function docstring."""
        model = nn.Sequential()
        for name, child in resnet.named_children():
            if name == 'conv1':
                assert hasattr(child, 'out_channels')
                first_conv = nn.Conv2d(in_channels=c, out_channels=child.out_channels,
                                       kernel_size=7, stride=2, padding=3, bias=False)
                nn.init.kaiming_normal_(first_conv.weight, mode='fan_out', nonlinearity='relu')
                model.add_module(name, first_conv)
            else:
                model.add_module(name, child)
        return model

    @property
    def out_channels(self) -> int:
        if self.name == 'resnet18':
            return 512
        elif self.name == 'resnet50':
            return 2048
        elif self.name == 'resnet101':
            return 2048
        else:
            raise KeyError

    @property
    def out_features(self) -> int:
        return self.out_channels
