import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet34_Weights, ResNet50_Weights, ResNet101_Weights


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, dilations=(6, 12, 18)):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_dil = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        3,
                        padding=dilation,
                        dilation=dilation,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
                for dilation in dilations
            ]
        )
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(
                out_channels * (2 + len(dilations)),
                out_channels,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs):
        height, width = inputs.shape[2:]
        features = [self.conv1(inputs)]
        features.extend(branch(inputs) for branch in self.conv_dil)
        pooled = self.global_pool(inputs)
        pooled = F.interpolate(
            pooled,
            size=(height, width),
            mode='bilinear',
            align_corners=False,
        )
        features.append(pooled)
        output = torch.cat(features, dim=1)
        return self.dropout(self.fuse(output))


class Decoder(nn.Module):
    def __init__(self, low_level_channels, num_classes):
        super().__init__()
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, inputs, low_level_features):
        inputs = F.interpolate(
            inputs,
            size=low_level_features.shape[2:],
            mode='bilinear',
            align_corners=False,
        )
        low_level_features = self.low_level_conv(low_level_features)
        inputs = torch.cat([inputs, low_level_features], dim=1)
        return self.classifier(self.fuse(inputs))


class DeepLabV3Plus(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone='resnet50',
        output_stride=16,
        pretrained=True,
    ):
        super().__init__()
        self.output_stride = output_stride

        if backbone == 'resnet50':
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet50(weights=weights)
            aspp_in_channels = 2048
            low_level_channels = 256
        elif backbone == 'resnet101':
            weights = ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet101(weights=weights)
            aspp_in_channels = 2048
            low_level_channels = 256
        elif backbone == 'resnet34':
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet34(weights=weights)
            aspp_in_channels = 512
            low_level_channels = 64
        else:
            raise ValueError(f'Unsupported backbone: {backbone}')

        if output_stride == 16:
            aspp_dilations = (6, 12, 18)
        elif output_stride == 8:
            aspp_dilations = (12, 24, 36)
        else:
            raise ValueError('output_stride must be 8 or 16.')

        self._build_backbone(resnet, output_stride)
        self.aspp = ASPP(aspp_in_channels, 256, aspp_dilations)
        self.decoder = Decoder(low_level_channels, num_classes)

    def _build_backbone(self, resnet, output_stride):
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        if output_stride == 16:
            self._make_dilated(self.layer3, dilation=2)
            self._make_dilated(self.layer4, dilation=4)
        else:
            self._make_dilated(self.layer2, dilation=2)
            self._make_dilated(self.layer3, dilation=4)
            self._make_dilated(self.layer4, dilation=8)

    @staticmethod
    def _make_dilated(layer, dilation):
        for block in layer:
            if hasattr(block, 'conv3'):
                block.conv2.stride = (1, 1)
                block.conv2.dilation = (dilation, dilation)
                block.conv2.padding = (dilation, dilation)
            else:
                block.conv1.stride = (1, 1)
                block.conv2.dilation = (dilation, dilation)
                block.conv2.padding = (dilation, dilation)

            if block.downsample is not None:
                for module in block.downsample:
                    if isinstance(module, nn.Conv2d):
                        module.stride = (1, 1)

    def forward(self, inputs):
        input_size = inputs.shape[2:]
        features = self.conv1(inputs)
        features = self.bn1(features)
        features = self.relu(features)
        features = self.maxpool(features)
        low_level_features = self.layer1(features)
        features = self.layer2(low_level_features)
        features = self.layer3(features)
        features = self.layer4(features)
        features = self.aspp(features)
        output = self.decoder(features, low_level_features)
        return F.interpolate(
            output,
            size=input_size,
            mode='bilinear',
            align_corners=False,
        )


if __name__ == '__main__':
    for backbone_name in ('resnet34', 'resnet50', 'resnet101'):
        network = DeepLabV3Plus(
            num_classes=2,
            backbone=backbone_name,
            output_stride=16,
            pretrained=False,
        )
        network.eval()
        sample = torch.randn(1, 3, 240, 320)
        with torch.inference_mode():
            prediction = network(sample)
        parameters = sum(parameter.numel() for parameter in network.parameters())
        print(
            f'{backbone_name}: input={tuple(sample.shape)}, '
            f'output={tuple(prediction.shape)}, parameters={parameters / 1e6:.2f} M'
        )
