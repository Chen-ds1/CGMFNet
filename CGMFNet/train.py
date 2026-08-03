import argparse
import os
import random
from collections import OrderedDict
from glob import glob

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm

import archs
import losses
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter, str2bool


ARCH_NAMES = list(archs.__all__)
LOSS_NAMES = list(losses.__all__) + ['BCEWithLogitsLoss']


def parse_args():
    parser = argparse.ArgumentParser(description='Train CGMFNet')
    parser.add_argument('--name', default='CGMFNet-L4', type=str)
    parser.add_argument('--epochs', default=100, type=int, metavar='N')
    parser.add_argument('-b', '--batch_size', default=8, type=int, metavar='N')
    parser.add_argument('-a', '--arch', default='CGMFNetL4', choices=ARCH_NAMES)
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int)
    parser.add_argument('--num_classes', default=1, type=int)
    parser.add_argument('--input_w', default=320, type=int)
    parser.add_argument('--input_h', default=240, type=int)
    parser.add_argument('--seed', default=41, type=int)
    parser.add_argument('--val_split', default=0.2, type=float)
    parser.add_argument('--iou_threshold', default=0.5, type=float)
    parser.add_argument('--loss', default='BCEDiceLoss', choices=LOSS_NAMES)
    parser.add_argument('--dataset', default='SeedingImg', type=str)
    parser.add_argument('--ext', default='.png', type=str)
    parser.add_argument('--rgb_folder', default='images', type=str)
    parser.add_argument('--depth_folder', default='depths', type=str)
    parser.add_argument('--rgb_mask_folder', default='rgb_masks', type=str)
    parser.add_argument('--depth_mask_folder', default='depth_masks', type=str)
    parser.add_argument('--optimizer', default='SGD', choices=['Adam', 'SGD'])
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--nesterov', default=False, type=str2bool)
    parser.add_argument(
        '--scheduler',
        default='CosineAnnealingLR',
        choices=[
            'CosineAnnealingLR',
            'ReduceLROnPlateau',
            'MultiStepLR',
            'ConstantLR',
        ],
    )
    parser.add_argument('--min_lr', default=1e-5, type=float)
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='30,60,90', type=str)
    parser.add_argument('--gamma', default=2 / 3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--model_dir', default='./models', type=str)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_loss_and_iou(config, model, rgb_input, depth_input, target, criterion):
    outputs = model(rgb_input, depth_input)
    if isinstance(outputs, (list, tuple)):
        loss = sum(criterion(output, target) for output in outputs) / len(outputs)
        metric_output = outputs[-1]
    else:
        loss = criterion(outputs, target)
        metric_output = outputs
    iou = iou_score(
        metric_output,
        target,
        threshold=config['iou_threshold'],
    )
    return loss, iou


def train_one_epoch(config, loader, model, criterion, optimizer, device):
    meters = {'loss': AverageMeter(), 'iou': AverageMeter()}
    model.train()

    progress = tqdm(loader, desc='Training')
    for rgb_input, rgb_target, depth_input, _, _ in progress:
        rgb_input = rgb_input.to(device, non_blocking=True)
        depth_input = depth_input.to(device, non_blocking=True)
        rgb_target = rgb_target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss, iou = calculate_loss_and_iou(
            config,
            model,
            rgb_input,
            depth_input,
            rgb_target,
            criterion,
        )
        loss.backward()
        optimizer.step()

        batch_size = rgb_input.size(0)
        meters['loss'].update(loss.item(), batch_size)
        meters['iou'].update(iou, batch_size)
        progress.set_postfix(
            loss=f"{meters['loss'].avg:.4f}",
            iou=f"{meters['iou'].avg:.4f}",
        )

    return OrderedDict(
        loss=meters['loss'].avg,
        iou=meters['iou'].avg,
    )


@torch.no_grad()
def validate(config, loader, model, criterion, device):
    meters = {'loss': AverageMeter(), 'iou': AverageMeter()}
    model.eval()

    progress = tqdm(loader, desc='Validating')
    for rgb_input, rgb_target, depth_input, _, _ in progress:
        rgb_input = rgb_input.to(device, non_blocking=True)
        depth_input = depth_input.to(device, non_blocking=True)
        rgb_target = rgb_target.to(device, non_blocking=True)

        loss, iou = calculate_loss_and_iou(
            config,
            model,
            rgb_input,
            depth_input,
            rgb_target,
            criterion,
        )
        batch_size = rgb_input.size(0)
        meters['loss'].update(loss.item(), batch_size)
        meters['iou'].update(iou, batch_size)
        progress.set_postfix(
            loss=f"{meters['loss'].avg:.4f}",
            iou=f"{meters['iou'].avg:.4f}",
        )

    return OrderedDict(
        loss=meters['loss'].avg,
        iou=meters['iou'].avg,
    )


def build_datasets(config):
    image_pattern = os.path.join(
        'inputs',
        config['dataset'],
        config['rgb_folder'],
        '*' + config['ext'],
    )
    image_ids = [
        os.path.splitext(os.path.basename(path))[0]
        for path in sorted(glob(image_pattern))
    ]
    if len(image_ids) < 2:
        raise RuntimeError(
            f'At least two RGB images are required. Found {len(image_ids)} '
            f'with pattern: {image_pattern}'
        )
    if not 0.0 < config['val_split'] < 1.0:
        raise ValueError('val_split must be between 0 and 1.')

    train_ids, val_ids = train_test_split(
        image_ids,
        test_size=config['val_split'],
        random_state=config['seed'],
    )
    additional_targets = {
        'depth_image': 'image',
        'depth_mask': 'mask',
    }
    train_transform = A.Compose(
        [
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Resize(config['input_h'], config['input_w']),
        ],
        additional_targets=additional_targets,
    )
    val_transform = A.Compose(
        [A.Resize(config['input_h'], config['input_w'])],
        additional_targets=additional_targets,
    )

    common_options = {
        'rgb_dir': os.path.join(
            'inputs', config['dataset'], config['rgb_folder']
        ),
        'depth_dir': os.path.join(
            'inputs', config['dataset'], config['depth_folder']
        ),
        'rgb_mask_dir': os.path.join(
            'inputs', config['dataset'], config['rgb_mask_folder']
        ),
        'depth_mask_dir': os.path.join(
            'inputs', config['dataset'], config['depth_mask_folder']
        ),
        'ext': config['ext'],
        'num_classes': config['num_classes'],
    }
    train_dataset = Dataset(
        img_ids=train_ids,
        transform=train_transform,
        **common_options,
    )
    val_dataset = Dataset(
        img_ids=val_ids,
        transform=val_transform,
        **common_options,
    )
    return train_dataset, val_dataset


def create_optimizer(config, parameters):
    if config['optimizer'] == 'Adam':
        return optim.Adam(
            parameters,
            lr=config['lr'],
            weight_decay=config['weight_decay'],
        )
    return optim.SGD(
        parameters,
        lr=config['lr'],
        momentum=config['momentum'],
        nesterov=config['nesterov'],
        weight_decay=config['weight_decay'],
    )


def create_scheduler(config, optimizer):
    if config['scheduler'] == 'CosineAnnealingLR':
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['epochs'],
            eta_min=config['min_lr'],
        )
    if config['scheduler'] == 'ReduceLROnPlateau':
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=config['factor'],
            patience=config['patience'],
            min_lr=config['min_lr'],
        )
    if config['scheduler'] == 'MultiStepLR':
        milestones = [int(epoch) for epoch in config['milestones'].split(',')]
        return lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=config['gamma'],
        )
    return None


def main(args):
    config = vars(args)
    set_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pin_memory = device.type == 'cuda'
    cudnn.benchmark = False
    cudnn.deterministic = True

    if not config['name']:
        supervision = 'DS' if config['deep_supervision'] else 'NODS'
        config['name'] = (
            f"{config['dataset']}_{config['arch']}_"
            f"{config['rgb_folder']}_{config['depth_folder']}_{supervision}"
        )

    model_output_dir = os.path.join(config['model_dir'], config['name'])
    os.makedirs(model_output_dir, exist_ok=True)
    with open(
        os.path.join(model_output_dir, 'config.yml'),
        'w',
        encoding='utf-8',
    ) as config_file:
        yaml.safe_dump(config, config_file, sort_keys=False)

    print(f'Device: {device}')
    for key, value in config.items():
        print(f'{key}: {value}')

    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().to(device)
    else:
        criterion = losses.__dict__[config['loss']]().to(device)

    model = archs.__dict__[config['arch']](
        config['num_classes'],
        config['input_channels'],
        config['deep_supervision'],
    ).to(device)
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = create_optimizer(config, parameters)
    scheduler = create_scheduler(config, optimizer)

    train_dataset, val_dataset = build_datasets(config)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=pin_memory,
        drop_last=False,
    )
    if len(train_loader) == 0:
        raise RuntimeError('The training loader is empty. Reduce batch_size.')

    log = OrderedDict(
        epoch=[],
        lr=[],
        loss=[],
        iou=[],
        val_loss=[],
        val_iou=[],
    )
    best_iou = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, config['epochs'] + 1):
        print(f"Epoch [{epoch}/{config['epochs']}]")
        train_log = train_one_epoch(
            config,
            train_loader,
            model,
            criterion,
            optimizer,
            device,
        )
        val_log = validate(config, val_loader, model, criterion, device)

        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_log['loss'])
        elif scheduler is not None:
            scheduler.step()

        print(
            f"loss {train_log['loss']:.4f} - iou {train_log['iou']:.4f} - "
            f"val_loss {val_log['loss']:.4f} - val_iou {val_log['iou']:.4f}"
        )
        log['epoch'].append(epoch)
        log['lr'].append(optimizer.param_groups[0]['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        pd.DataFrame(log).to_csv(
            os.path.join(model_output_dir, 'log.csv'),
            index=False,
        )

        epochs_without_improvement += 1
        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), os.path.join(model_output_dir, 'model.pth'))
            best_iou = val_log['iou']
            epochs_without_improvement = 0
            print(f'Saved best model with IoU {best_iou:.4f}.')

        if (
            config['early_stopping'] >= 0
            and epochs_without_improvement >= config['early_stopping']
        ):
            print('Early stopping.')
            break

        if device.type == 'cuda':
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main(parse_args())
