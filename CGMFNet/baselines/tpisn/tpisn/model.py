import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v3_small


def conv(cin, cout, kernel=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel, padding=dilation * (kernel // 2),
                  dilation=dilation, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class SequentialPSA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = channels // 2
        self.channel_q = nn.Conv2d(channels, 1, 1)
        self.channel_v = nn.Conv2d(channels, hidden, 1)
        self.channel_z = nn.Conv2d(hidden, channels, 1)
        self.norm = nn.LayerNorm([channels, 1, 1])
        self.spatial_q = nn.Conv2d(channels, hidden, 1)
        self.spatial_v = nn.Conv2d(channels, hidden, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.channel_q(x).flatten(2).softmax(-1)
        v = self.channel_v(x).flatten(2)
        context = torch.bmm(v, q.transpose(1, 2)).unsqueeze(-1)
        x = x * self.norm(self.channel_z(context)).sigmoid()
        q = self.spatial_q(x).mean((2, 3)).unsqueeze(1).softmax(-1)
        v = self.spatial_v(x).flatten(2)
        attention = torch.bmm(q, v).reshape(b, 1, h, w).sigmoid()
        return x * attention


class ASPP(nn.Module):
    def __init__(self, cin, channels):
        super().__init__()
        self.branches = nn.ModuleList([conv(cin, channels)] +
                                     [conv(cin, channels, 3, r) for r in (6, 12, 18)])

        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  nn.Conv2d(cin, channels, 1), nn.ReLU(inplace=True))
        self.project = conv(channels * 5, channels)

    def forward(self, x):
        pooled = F.interpolate(self.pool(x), size=x.shape[-2:], mode='bilinear', align_corners=False)
        return self.project(torch.cat([f(x) for f in self.branches] + [pooled], 1))


class Decoder(nn.Module):
    def __init__(self, channels, outputs, dropout):
        super().__init__()
        self.low_project = conv(16, 48)
        self.decode = nn.Sequential(nn.Dropout2d(dropout),
                                    conv(channels + 48, channels, 3),
                                    nn.Conv2d(channels, outputs, 1))

    def forward(self, low, high, size):
        high = F.interpolate(high, size=low.shape[-2:], mode='bilinear', align_corners=False)
        x = self.decode(torch.cat([self.low_project(low), high], 1))
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class TPISN(nn.Module):
    def __init__(self, embedding_dim=4, channels=128, dropout=0.1):
        super().__init__()
        self.config = dict(embedding_dim=embedding_dim, channels=channels, dropout=dropout)

        self.backbone = mobilenet_v3_small(weights=None).features[:8]
        self.context = nn.Sequential(ASPP(48, channels), SequentialPSA(channels))
        self.segmentation = Decoder(channels, 1, dropout)
        self.embedding = Decoder(channels, embedding_dim, dropout)

    def forward(self, x):
        size = x.shape[-2:]
        low = None
        for i, block in enumerate(self.backbone):
            x = block(x)
            if i == 1:
                low = x
        high = self.context(x)
        return {'logits': self.segmentation(low, high, size),
                'embeddings': self.embedding(low, high, size)}
