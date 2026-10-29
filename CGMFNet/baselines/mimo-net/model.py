import torch
from torch import nn
from torch.nn import functional as F
import timm

def resize(x, ref):
    return F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)

class Flow(nn.Sequential):
    def __init__(self, a, b):
        super().__init__(nn.Conv2d(a,b,3,padding=1,bias=False), nn.SiLU(),
                         nn.BatchNorm2d(b), nn.Dropout2d(0.1))

class WeightedFusion(nn.Module):
    def __init__(self, n, c):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n,c,1,1)/n)
        self.conv = nn.Sequential(nn.Conv2d(c,c,3,padding=1,bias=False),
                                  nn.BatchNorm2d(c),nn.SiLU())
    def forward(self, xs):

        return self.conv(sum(self.weights[i]*resize(x,xs[0]) for i,x in enumerate(xs)))

class DFF(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.top = nn.ModuleList([WeightedFusion(2,c) for _ in range(4)])
        self.bottom = nn.ModuleList([WeightedFusion(3 if i<4 else 2,c) for i in range(1,5)])
    def forward(self,x):
        td=list(x)
        for i in range(3,-1,-1): td[i]=self.top[i]([x[i],td[i+1]])
        out=[td[0]]
        for i in range(1,5):
            xs=[x[i],td[i],out[i-1]] if i<4 else [x[i],out[i-1]]
            out.append(self.bottom[i-1](xs))
        return out

class Reweight(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.dw=nn.Sequential(nn.Conv2d(c,c,3,padding=1,groups=c,bias=False),nn.BatchNorm2d(c),nn.SiLU())
        self.se=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(c,max(c//4,1),1),nn.SiLU(),nn.Conv2d(max(c//4,1),c,1),nn.Sigmoid())
    def forward(self,x):
        x=self.dw(x)
        return x*self.se(x)

class FF(nn.Module):
    def __init__(self,n,c):
        super().__init__()
        self.r=nn.ModuleList([Reweight(c) for _ in range(n)])
        self.out=nn.Sequential(nn.Conv2d(c,c,3,padding=1,groups=c,bias=False),nn.BatchNorm2d(c),nn.SiLU())
    def forward(self,xs):
        return self.out(sum(r(resize(x,xs[0])) for r,x in zip(self.r,xs)))

class MIMONet(nn.Module):
    def __init__(self,backbone='maxvit_tiny_tf_224',width=64,image_hw=(384,768),pretrained=False,depth_mode='pseudocolor'):
        super().__init__()
        args=dict(pretrained=pretrained,features_only=True,img_size=tuple(image_hw))
        self.rgb=timm.create_model(backbone,in_chans=3,**args)
        self.depth=timm.create_model(backbone,in_chans=3 if depth_mode=='pseudocolor' else 1,**args)
        channels=self.rgb.feature_info.channels()
        if len(channels)!=5: raise ValueError('Five encoder scales required')
        self.flow=nn.ModuleList([Flow(c,width) for c in channels])
        self.dff=nn.Sequential(DFF(width),DFF(width))
        self.seg=nn.ModuleList([FF(3,width) for _ in range(4)])
        self.line=nn.ModuleList([FF(3,width) for _ in range(4)])
        self.seg_head=nn.Conv2d(width,1,1)
        self.line_head=nn.Conv2d(width,1,1)
    def forward(self,rgb,depth):
        x=self.dff([f(a+b) for f,a,b in zip(self.flow,self.rgb(rgb),self.depth(depth))])
        s=l=x[-1]
        for i in range(3,-1,-1):

            s,l=self.seg[i]([x[i],s,l]),self.line[i]([x[i],l,s])
        size=rgb.shape[-2:]
        return {'plant':F.interpolate(self.seg_head(s),size=size,mode='bilinear',align_corners=False),
                'line':F.interpolate(self.line_head(l),size=size,mode='bilinear',align_corners=False)}
