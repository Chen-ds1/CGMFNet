import argparse

import torch
from fvcore.nn import FlopCountAnalysis

from pspnet.pspnet import PSPNet


def parse_args():
    parser = argparse.ArgumentParser(description='Report PSPNet model complexity')
    parser.add_argument('--input_height', default=240, type=int)
    parser.add_argument('--input_width', default=320, type=int)
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument(
        '--backbone',
        default='resnet34',
        choices=['resnet34', 'resnet50', 'resnet101'],
    )
    return parser.parse_args()


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PSPNet(
        num_classes=args.num_classes,
        backbone=args.backbone,
        pretrained=False,
        use_aux=True,
    ).to(device)
    model.eval()

    sample = torch.randn(
        1,
        3,
        args.input_height,
        args.input_width,
        device=device,
    )
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    flops = FlopCountAnalysis(model, sample).total()

    print(f'Device: {device}')
    print(f'Backbone: {args.backbone}')
    print(f'Input size: {args.input_width} x {args.input_height}')
    print(f'Trainable parameters: {trainable_parameters / 1e6:.2f} M')
    print(f'Total parameters: {total_parameters / 1e6:.2f} M')
    print(f'FLOPs: {flops / 1e9:.2f} G')


if __name__ == '__main__':
    main(parse_args())
