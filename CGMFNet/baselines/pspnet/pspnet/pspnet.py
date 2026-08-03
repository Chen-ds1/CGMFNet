import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet34_Weights, ResNet50_Weights, ResNet101_Weights


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        return self.block(inputs)


class ConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        return self.block(inputs)


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, out_channels=512, bin_sizes=(1, 2, 3, 6)):
        super().__init__()
        intermediate_channels = out_channels // 4

        self.stages = nn.ModuleList()
        for bin_size in bin_sizes:
            convolution = (
                ConvReLU(in_channels, intermediate_channels, kernel_size=1)
                if bin_size == 1
                else ConvBNReLU(in_channels, intermediate_channels, kernel_size=1)
            )
            self.stages.append(
                nn.Sequential(nn.AdaptiveAvgPool2d(bin_size), convolution)
            )

        self.self_conv = ConvBNReLU(
            in_channels,
            intermediate_channels,
            kernel_size=1,
        )
        fused_channels = intermediate_channels * (len(bin_sizes) + 1)
        self.fuse = nn.Sequential(
            ConvBNReLU(fused_channels, out_channels, kernel_size=3),
            nn.Dropout(0.1),
        )

    def forward(self, inputs):
        height, width = inputs.shape[2:]
        pooled_features = []
        for stage in self.stages:
            feature = stage(inputs)
            feature = F.interpolate(
                feature,
                size=(height, width),
                mode='bilinear',
                align_corners=False,
            )
            pooled_features.append(feature)

        projected_input = self.self_conv(inputs)
        output = torch.cat([projected_input] + pooled_features, dim=1)
        return self.fuse(output)


class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            ConvBNReLU(in_channels, 256, kernel_size=3),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(self, inputs):
        return self.classifier(inputs)


class PSPNet(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone='resnet50',
        pretrained=True,
        use_aux=True,
    ):
        super().__init__()
        self.use_aux = use_aux

        if backbone == 'resnet50':
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet50(weights=weights)
            feature_channels = [256, 512, 1024, 2048]
        elif backbone == 'resnet101':
            weights = ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet101(weights=weights)
            feature_channels = [256, 512, 1024, 2048]
        elif backbone == 'resnet34':
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet34(weights=weights)
            feature_channels = [64, 128, 256, 512]
        else:
            raise ValueError(f'Unsupported backbone: {backbone}')

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self._make_dilated(self.layer3, dilation=2)
        self._make_dilated(self.layer4, dilation=4)

        self.ppm = PyramidPoolingModule(feature_channels[3], out_channels=512)
        self.classifier = nn.Sequential(
            ConvBNReLU(512, 512, kernel_size=3),
            nn.Dropout(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1),
        )

        if use_aux:
            self.aux_classifier = AuxiliaryClassifier(
                feature_channels[2],
                num_classes,
            )

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
        features = self.layer1(features)
        features = self.layer2(features)
        auxiliary_features = self.layer3(features)
        features = self.layer4(auxiliary_features)
        features = self.ppm(features)

        output = self.classifier(features)
        output = F.interpolate(
            output,
            size=input_size,
            mode='bilinear',
            align_corners=False,
        )

        if self.use_aux and self.training:
            auxiliary_output = self.aux_classifier(auxiliary_features)
            auxiliary_output = F.interpolate(
                auxiliary_output,
                size=input_size,
                mode='bilinear',
                align_corners=False,
            )
            return output, auxiliary_output
        return output


if __name__ == '__main__':
    network = PSPNet(
        num_classes=2,
        backbone='resnet34',
        pretrained=False,
    )
    network.train()
    sample = torch.randn(1, 3, 240, 320)
    prediction = network(sample)
    if isinstance(prediction, tuple):
        print(f'Main output: {prediction[0].shape}')
        print(f'Auxiliary output: {prediction[1].shape}')
    else:
        print(f'Output: {prediction.shape}')
    parameters = sum(parameter.numel() for parameter in network.parameters())
    print(f'Parameters: {parameters / 1e6:.2f} M')
