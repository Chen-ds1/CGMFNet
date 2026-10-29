# ESANet: Copyright 2021 Neuroinformatics and Cognitive Robotics Lab,
# Technische Universitaet Ilmenau. See official_esanet/LICENSE.
# DFormerv2/Ham: Copyright (c) 2023 VCIP-RGBD. See DFORMER_LICENSE.
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).parent / 'official_esanet'))
from src.models.model import ESANet, Decoder
from src.models.resnet import ResNet34
from src.models.context_modules import get_context_module
from src.models.model_utils import ConvBNAct, SqueezeAndExcitation


class DepthOnlyESANet(nn.Module):
    def __init__(self, height, width):
        super().__init__()
        act = nn.ReLU(inplace=True)
        self.encoder = ResNet34(block='NonBottleneck1D', pretrained_on_imagenet=False,
                                input_channels=3, activation=act)
        self.se = nn.ModuleList([SqueezeAndExcitation(c, activation=act) for c in (64,64,128,256,512)])
        self.skip = nn.ModuleList([ConvBNAct(a,b,1,activation=act) for a,b in [(64,128),(128,256),(256,512)]])
        self.context, channels = get_context_module('ppm',512,512,(height//32,width//32),act,'bilinear')
        self.decoder = Decoder(channels,[512,256,128],act,[3,3,3],'add','bilinear',2)

    def forward(self, depth_color):
        x = F.max_pool2d(self.se[0](self.encoder.forward_first_conv(depth_color)),3,2,1)
        skips = []
        for i in range(4):
            x = self.se[i+1](getattr(self.encoder, f'forward_layer{i+1}')(x))
            if i < 3:
                skips.append(self.skip[i](x))
        return self.decoder([self.context(x), *reversed(skips)])


class ConvNormAct(nn.Sequential):
    def __init__(self, cin, cout, norm=True, activation=True):
        layers=[nn.Conv2d(cin,cout,1,bias=not norm)]
        if norm:
            layers.append(nn.BatchNorm2d(cout))
        if activation:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class NMF(nn.Module):
    def forward(self,x):
        b,c,h,w=x.shape
        x=x.reshape(b,c,h*w)
        bases=F.normalize(torch.rand(b,c,64,device=x.device,dtype=x.dtype),dim=1)
        coef=F.softmax(x.transpose(1,2).bmm(bases),dim=-1)
        for _ in range(6 if self.training else 7):
            coef=coef*x.transpose(1,2).bmm(bases)/(coef.bmm(bases.transpose(1,2).bmm(bases))+1e-6)
            bases=bases*x.bmm(coef)/(bases.bmm(coef.transpose(1,2).bmm(coef))+1e-6)
        coef=coef*x.transpose(1,2).bmm(bases)/(coef.bmm(bases.transpose(1,2).bmm(bases))+1e-6)
        return bases.bmm(coef.transpose(1,2)).reshape(b,c,h,w)


class LightHamHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.squeeze=ConvNormAct(128+256+512,512)
        self.ham_in=ConvNormAct(512,512,norm=False,activation=False)
        self.nmf=NMF()
        self.ham_out=ConvNormAct(512,512,activation=False)
        self.align=ConvNormAct(512,512)
        self.dropout=nn.Dropout2d(.1)
        self.classifier=nn.Conv2d(512,2,1)

    def forward(self,features):
        selected=features[1:]
        xs=[F.interpolate(x,selected[0].shape[-2:],mode='bilinear',align_corners=False) for x in selected]
        x=self.squeeze(torch.cat(xs,dim=1))

        with torch.autocast(device_type=x.device.type,enabled=False):
            y=self.nmf(F.relu(self.ham_in(x.float()),inplace=True))
            y=self.ham_out(y)
        x=F.relu(x+y,inplace=True)
        return self.classifier(self.dropout(self.align(x)))


class PseudocolorDFormerv2(nn.Module):
    def __init__(self,drop_path_rate=.25):
        super().__init__()
        from dformerv2_encoder import DFormerv2_S
        self.color_guide=nn.Sequential(nn.Conv2d(3,16,1),nn.GELU(),nn.Conv2d(16,1,1),nn.Sigmoid())
        self.backbone=DFormerv2_S(drop_path_rate=drop_path_rate)
        self.decode_head=LightHamHead()

    def forward(self,rgb,depth_color):
        guide=self.color_guide(depth_color)
        features=self.backbone(rgb,guide)
        return F.interpolate(self.decode_head(features),rgb.shape[-2:],mode='bilinear',align_corners=False)


def create(c):
    if c['model']=='esanet_depth':
        return DepthOnlyESANet(c['height'],c['width'])
    if c['model']=='esanet_rgbd':
        model=ESANet(height=c['height'],width=c['width'],num_classes=2,
                     encoder_rgb='resnet34',encoder_depth='resnet34',encoder_block='NonBottleneck1D',
                     channels_decoder=[512,256,128],nr_decoder_blocks=[3,3,3],pretrained_on_imagenet=False,
                     context_module='ppm',upsampling='bilinear',fuse_depth_in_rgb_encoder='SE-add')

        old=model.encoder_depth.conv1
        model.encoder_depth.conv1=nn.Conv2d(3,old.out_channels,old.kernel_size,old.stride,old.padding,bias=False)
        nn.init.kaiming_normal_(model.encoder_depth.conv1.weight,mode='fan_out',nonlinearity='relu')
        return model
    if c['model']=='dformerv2':
        return PseudocolorDFormerv2(c['architecture']['drop_path_rate'])
    raise ValueError(c['model'])
