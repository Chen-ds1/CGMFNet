import math
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

class RCASC(nn.Module):

    def __init__(self, c, pretrained=False):
        super().__init__()
        self.c = c
        b = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.backbone = nn.Sequential(*list(b.children())[:-2])
        self.reduce = nn.Conv2d(512, c['pool_channels'], 1)
        size = c['pool_channels'] * math.ceil(c['height'] / 32) * math.ceil(c['width'] / 32)
        self.shared = nn.Sequential(nn.Flatten(), nn.Linear(size, c['hidden']), nn.ReLU())
        n = 2 * c['max_rows'] * c['anchors']
        self.location = nn.Linear(c['hidden'], n * c['bins'])
        self.existence = nn.Linear(c['hidden'], n * 2)

    def forward(self, x):
        z = self.shared(self.reduce(self.backbone(x)))
        c = self.c
        shape = (len(x), 2, c['max_rows'], c['anchors'])
        return (self.location(z).reshape(*shape, c['bins']), self.existence(z).reshape(*shape, 2))
