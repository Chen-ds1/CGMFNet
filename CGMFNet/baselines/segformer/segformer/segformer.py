import torch
import torch.nn as nn
import torch.nn.functional as F

from segformer.mit import MiT


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        return self.relu(self.norm(self.conv(inputs)))


class SegFormerHead(nn.Module):
    def __init__(self, in_channels, embed_dim=256, num_classes=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp_layers = nn.ModuleList(
            [MLP(channels, embed_dim) for channels in in_channels]
        )
        self.fuse = nn.Sequential(
            MLP(embed_dim * len(in_channels), embed_dim),
            nn.Conv2d(embed_dim, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(embed_dim, num_classes, 1)

    def forward(self, features):
        projected_features = [
            projection(feature)
            for feature, projection in zip(features, self.mlp_layers)
        ]
        target_size = projected_features[0].shape[2:]
        resized_features = [projected_features[0]]
        resized_features.extend(
            F.interpolate(
                feature,
                size=target_size,
                mode='bilinear',
                align_corners=False,
            )
            for feature in projected_features[1:]
        )
        fused = torch.cat(resized_features, dim=1)
        return self.classifier(self.fuse(fused))


class SegFormer(nn.Module):
    CONFIGS = {
        'B0': {
            'embed_dims': [32, 64, 160, 256],
            'num_heads': [1, 2, 5, 8],
            'depths': [2, 2, 2, 2],
            'sr_ratios': [8, 4, 2, 1],
            'mlp_ratios': [8, 8, 4, 4],
            'decoder_dim': 256,
        },
        'B1': {
            'embed_dims': [64, 128, 320, 512],
            'num_heads': [1, 2, 5, 8],
            'depths': [2, 2, 2, 2],
            'sr_ratios': [8, 4, 2, 1],
            'mlp_ratios': [8, 8, 4, 4],
            'decoder_dim': 256,
        },
        'B2': {
            'embed_dims': [64, 128, 320, 512],
            'num_heads': [1, 2, 5, 8],
            'depths': [3, 3, 6, 3],
            'sr_ratios': [8, 4, 2, 1],
            'mlp_ratios': [8, 8, 4, 4],
            'decoder_dim': 768,
        },
        'B3': {
            'embed_dims': [64, 128, 320, 512],
            'num_heads': [1, 2, 5, 8],
            'depths': [3, 3, 18, 3],
            'sr_ratios': [8, 4, 2, 1],
            'mlp_ratios': [8, 8, 4, 4],
            'decoder_dim': 768,
        },
        'B4': {
            'embed_dims': [64, 128, 320, 512],
            'num_heads': [1, 2, 5, 8],
            'depths': [3, 8, 27, 3],
            'sr_ratios': [8, 4, 2, 1],
            'mlp_ratios': [8, 8, 4, 4],
            'decoder_dim': 768,
        },
        'B5': {
            'embed_dims': [64, 128, 320, 512],
            'num_heads': [1, 2, 5, 8],
            'depths': [3, 6, 40, 3],
            'sr_ratios': [8, 4, 2, 1],
            'mlp_ratios': [8, 8, 4, 4],
            'decoder_dim': 768,
        },
    }

    def __init__(
        self,
        num_classes=2,
        backbone='B0',
        in_channels=3,
        pretrained=False,
    ):
        super().__init__()
        if backbone not in self.CONFIGS:
            choices = ', '.join(self.CONFIGS)
            raise ValueError(f'Unsupported backbone: {backbone}. Choose from {choices}.')
        if pretrained:
            raise NotImplementedError(
                'Automatic pretrained MiT weights are not included. '
                'Load external weights explicitly before training.'
            )

        config = self.CONFIGS[backbone]
        self.encoder = MiT(
            in_channels=in_channels,
            embed_dims=config['embed_dims'],
            num_heads=config['num_heads'],
            depths=config['depths'],
            sr_ratios=config['sr_ratios'],
            mlp_ratios=config['mlp_ratios'],
            drop_rate=0.0,
            drop_path_rate=0.1,
        )
        self.decoder = SegFormerHead(
            in_channels=config['embed_dims'],
            embed_dim=config['decoder_dim'],
            num_classes=num_classes,
        )

    def forward(self, inputs):
        input_size = inputs.shape[2:]
        features = self.encoder(inputs)
        output = self.decoder(features)
        return F.interpolate(
            output,
            size=input_size,
            mode='bilinear',
            align_corners=False,
        )


if __name__ == '__main__':
    network = SegFormer(num_classes=2, backbone='B0')
    network.eval()
    sample = torch.randn(1, 3, 240, 320)
    with torch.inference_mode():
        prediction = network(sample)
    parameters = sum(parameter.numel() for parameter in network.parameters())
    print(f'Input: {tuple(sample.shape)}')
    print(f'Output: {tuple(prediction.shape)}')
    print(f'Parameters: {parameters / 1e6:.2f} M')
