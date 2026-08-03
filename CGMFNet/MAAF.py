import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):

    def __init__(self, in_ch, out_ch, r=16):
        super().__init__()
        hidden = max(in_ch // r, 4)
        self.fc1 = nn.Conv2d(in_ch, hidden, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, out_ch, 1, bias=False)

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1)
        max_ = F.adaptive_max_pool2d(x, 1)
        w = self.fc2(self.relu(self.fc1(avg + max_)))
        return torch.sigmoid(w)

class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=5, dilation=3):
        super().__init__()
        padding = ((kernel_size - 1) * dilation) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding,
                              dilation=dilation, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))
        return scale

class DRSMC(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(channels, channels, 3, 3))
        self.bias = nn.Parameter(torch.zeros(channels))

        self.scale_weights = nn.Parameter(torch.ones(3) / 3)

    def forward(self, x):

        out1 = F.conv2d(x, self.weight, self.bias, padding=1, dilation=1)

        out2 = F.conv2d(x, self.weight, self.bias, padding=2, dilation=2)

        out3 = F.conv2d(x, self.weight, self.bias, padding=4, dilation=4)

        w = F.softmax(self.scale_weights, dim=0)
        return w[0] * out1 + w[1] * out2 + w[2] * out3

class MAAF(nn.Module):

    def __init__(self, in_channels, out_channels, in_channels_list, factor=4.0):

        super().__init__()

        n = len(in_channels_list)
        self.input_weights = nn.Parameter(torch.ones(n) / n)

        mid_channels = max(int(out_channels // factor), 8)

        self.down = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.multi_scale = DRSMC(mid_channels)

        self.channel_attention = ChannelAttention(mid_channels, mid_channels, r=16)
        self.spatial_attention  = SpatialAttention(kernel_size=5, dilation=3)

        self.up = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, *inputs):

        fused = torch.cat([w * x for w, x in zip(self.input_weights, inputs)], dim=1)

        x_down = self.down(fused)

        ch_att = self.channel_attention(x_down)
        x_c = x_down * ch_att

        x_ms = self.multi_scale(x_down)
        sp_att = self.spatial_attention(x_ms)
        x_s = x_ms * sp_att

        x_fused = x_c + x_s

        x_up = self.up(x_fused)
        shortcut = self.shortcut(fused)
        return F.relu(x_up + shortcut, inplace=True)

if __name__ == '__main__':
    x1 = torch.randn(2, 32, 96, 96)
    x2 = torch.randn(2, 64, 96, 96)
    model = MAAF(in_channels=96, out_channels=32, in_channels_list=[32, 64])
    out = model(x1, x2)
    print(f"Output shape: {out.shape}")
