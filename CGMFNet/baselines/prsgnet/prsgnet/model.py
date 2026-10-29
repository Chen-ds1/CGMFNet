import math
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, ResNet18_Weights


def initial_priors():

    rows = [[x, 0., math.radians(a)] for x in torch.linspace(0, 1, 36).tolist()
            for a in (36, 72, 108, 144)]
    rows += [[0., y, math.radians(a)] for y in torch.linspace(0, .5, 12).tolist()
             for a in (30, 60)]
    rows += [[1., y, math.radians(a)] for y in torch.linspace(0, .5, 12).tolist()
             for a in (120, 150)]
    return torch.tensor(rows)


def line_x(geometry, n, height, width):
    y = torch.linspace(0, 1, n, device=geometry.device)
    theta = geometry[..., 2].clamp(.01, math.pi - .01)
    return geometry[..., 0, None] + ((y - geometry[..., 1, None]) *
            (height - 1) / (width - 1) / torch.tan(theta[..., None]))


class RFEM(nn.Module):
    def __init__(self, c=64, n=36):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(c, c, 3, padding=1), nn.ReLU(),
                                  nn.Conv1d(c, c, 3, padding=1), nn.ReLU())
        self.fc = nn.Linear(c * n, c)

    def forward(self, feature, xs):
        b, p, n = xs.shape
        y = torch.linspace(1, -1, n, device=xs.device).expand(b, p, n)
        grid = torch.stack((2 * xs - 1, y), -1)
        roi = F.grid_sample(feature, grid, align_corners=True, padding_mode='zeros')
        roi = roi.permute(0, 2, 1, 3).reshape(b * p, 64, n)
        local = self.fc(self.conv(roi).flatten(1)).reshape(b, p, 64)
        global_features = F.interpolate(feature, (10, 25), mode='bilinear',
                                        align_corners=False).flatten(2)
        attention = (local @ global_features / math.sqrt(64)).softmax(-1)
        return local + attention @ global_features.transpose(1, 2)


class PRSGNet(nn.Module):
    def __init__(self, n=36, pretrained=False):
        super().__init__()
        self.n = n
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.backbone.fc = nn.Identity()
        self.lateral = nn.ModuleList(nn.Conv2d(c, 64, 1) for c in (128, 256, 512))
        self.smooth = nn.ModuleList(nn.Conv2d(64, 64, 3, padding=1) for _ in range(3))
        self.rfem = nn.ModuleList(RFEM(n=n) for _ in range(3))
        self.cls = nn.ModuleList(nn.Linear(64, 2) for _ in range(3))
        self.reg = nn.ModuleList(nn.Linear(64, n + 4) for _ in range(3))
        self.register_buffer('priors', initial_priors())
        self.seg = nn.Sequential(nn.Conv2d(192, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 1, 1))
        self.ipm = nn.Sequential(nn.Conv2d(192, 64, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(64, 32, 3, stride=2, padding=1), nn.ReLU(),
                                 nn.AdaptiveAvgPool2d((4, 8)), nn.Flatten(),
                                 nn.Linear(32 * 4 * 8, 64), nn.ReLU(), nn.Linear(64, 6))
        for layer in self.reg:
            nn.init.normal_(layer.weight, std=.001)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.ipm[-1].weight)
        nn.init.zeros_(self.ipm[-1].bias)

    def forward(self, image, auxiliary=None):
        if auxiliary is None:
            auxiliary = self.training
        b, _, h, w = image.shape
        net = self.backbone
        x = net.maxpool(net.relu(net.bn1(net.conv1(image))))
        x = net.layer1(x)
        c3 = net.layer2(x)
        c4 = net.layer3(c3)
        c5 = net.layer4(c4)
        levels = [layer(c) for layer, c in zip(self.lateral, (c3, c4, c5))]
        for i in (1, 0):
            levels[i] = levels[i] + F.interpolate(levels[i+1], levels[i].shape[-2:], mode='nearest')
        levels = [conv(f) for conv, f in zip(self.smooth, levels)][::-1]
        geometry = self.priors.unsqueeze(0).expand(b, -1, -1)
        stages = []
        for feature, rfem, cls, reg in zip(levels, self.rfem, self.cls, self.reg):
            embedding = rfem(feature, line_x(geometry, self.n, h, w))
            delta = reg(embedding)
            geometry = geometry + delta[..., :3]
            geometry = torch.cat((geometry[..., :2].clamp(0, 1),
                                   geometry[..., 2:3].clamp(.01, math.pi-.01)), -1)
            xs = line_x(geometry, self.n, h, w) + delta[..., 4:]
            stages.append({'logits': cls(embedding), 'geometry': geometry,
                           'length': delta[..., 3].sigmoid(), 'xs': xs})

        result = {'stages': stages}
        if auxiliary:
            size = levels[-1].shape[-2:]
            fused = torch.cat([F.interpolate(f, size, mode='bilinear', align_corners=False)
                               for f in levels], 1)
            result['seg'] = F.interpolate(self.seg(fused), (h, w), mode='bilinear', align_corners=False)
            z = self.ipm(fused)

            a, bb, c, d, e, f = z.unbind(-1)
            a, d = a.clamp(-2, 2).exp(), d.clamp(-2, 2).exp()
            e, f = .25 * e.tanh(), .5 * f.tanh()
            zero, one = torch.zeros_like(a), torch.ones_like(a)
            result['H'] = torch.stack((a, bb.tanh(), c.tanh(), zero, d+e*f, e, zero, f, one), -1).reshape(b, 3, 3)
        return result
