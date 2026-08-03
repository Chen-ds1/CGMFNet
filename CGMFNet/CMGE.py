import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBottleneckBlock(nn.Module):

    def __init__(self, in_channels, out_channels, reduction=2):
        super().__init__()
        mid_channels = max(out_channels // reduction, 4)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.relu(out + residual)
        return out

class ChannelAttention(nn.Module):

    def __init__(self, in_ch, out_ch, r=1):
        super().__init__()
        hidden = max(in_ch // r, 4)
        self.fc1 = nn.Conv2d(in_ch, hidden, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, out_ch, 1, bias=False)

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1)
        max = F.adaptive_max_pool2d(x, 1)
        w = self.fc2(self.relu(self.fc1(avg + max)))
        return torch.sigmoid(w)

class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=5, dilation=3):
        super().__init__()
        padding = ((kernel_size - 1) * dilation) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding,
                              dilation=dilation, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.sigmoid(self.conv(torch.cat([avg, max], dim=1)))
        return scale

class SAGU(nn.Module):

    def __init__(self, in_planes, r=1):
        super().__init__()
        self.channel_att = ChannelAttention(in_planes * 2, in_planes, r)
        self.spatial_att = SpatialAttention(kernel_size=5, dilation=3)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, guide, main):
        combined = torch.cat([guide, main], dim=1)
        ch_w = self.channel_att(combined)
        sp_w = self.spatial_att(combined)
        weight = ch_w * sp_w

        similarity = F.cosine_similarity(guide, main, dim=1, eps=1e-8)
        similarity = similarity.unsqueeze(1)

        gate = torch.sigmoid(self.alpha) * weight * (1 - similarity.detach())

        return (1 - gate) * main + gate * guide

class CMGE(nn.Module):

    def __init__(self, in_channels, out_channels, middle_channels=None, r=1):
        super().__init__()
        if middle_channels is None:
            middle_channels = out_channels

        self.rgb_encoder = ResidualBottleneckBlock(in_channels, out_channels, reduction=2)
        self.depth_encoder = ResidualBottleneckBlock(in_channels, out_channels, reduction=2)
        self.sagu = SAGU(out_channels, r)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        rgb, depth = x

        feat_rgb = self.rgb_encoder(rgb)
        feat_depth = self.depth_encoder(depth)

        rec_rgb = self.sagu(feat_depth, feat_rgb)
        rec_depth = self.sagu(feat_rgb, feat_depth)

        rgb_out = self.relu((feat_rgb + rec_rgb) / 2)
        depth_out = self.relu((feat_depth + rec_depth) / 2)

        fused_avg = (rgb_out + depth_out) / 2

        return [rgb_out, depth_out], fused_avg

if __name__ == '__main__':

    rgb_img = torch.randn(8, 3, 96, 96)
    depth_img = torch.randn(8, 3, 96, 96)

    model = CMGE(in_channels=3, out_channels=64, r=1)
    [rgb_enhanced, depth_enhanced], fused = model([rgb_img, depth_img])

    print("Enhanced RGB shape:", rgb_enhanced.shape)
    print("Enhanced Depth shape:", depth_enhanced.shape)
    print("Fused average shape:", fused.shape)
