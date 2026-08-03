import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SegmentationDataset
from deeplabv3plus.deeplabv3plus import DeepLabV3Plus


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {'true', '1', 'yes', 'y'}:
        return True
    if value in {'false', '0', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError('Expected a boolean value.')


def parse_args():
    parser = argparse.ArgumentParser(description='Train DeepLabV3+')
    parser.add_argument('--image_dir', default='./inputs/images', type=str)
    parser.add_argument('--mask_dir', default='./inputs/masks', type=str)
    parser.add_argument('--image_suffix', default='.png', type=str)
    parser.add_argument('--mask_suffix', default='.png', type=str)
    parser.add_argument('--input_height', default=240, type=int)
    parser.add_argument('--input_width', default=320, type=int)
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--ignore_index', default=None, type=int)
    parser.add_argument(
        '--backbone',
        default='resnet34',
        choices=['resnet34', 'resnet50', 'resnet101'],
    )
    parser.add_argument('--pretrained', default=True, type=str2bool)
    parser.add_argument('--output_stride', default=16, type=int, choices=[8, 16])
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--val_split', default=0.1, type=float)
    parser.add_argument('--use_dice', default=False, type=str2bool)
    parser.add_argument('--dice_weight', default=0.5, type=float)
    parser.add_argument('--amp', default=True, type=str2bool)
    parser.add_argument('--grad_accum', default=1, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--seed', default=41, type=int)
    parser.add_argument('--save_dir', default='./models', type=str)
    parser.add_argument('--save_every', default=20, type=int)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_sample_ids(sample_ids, val_split, seed):
    if not 0.0 < val_split < 1.0:
        raise ValueError('val_split must be between 0 and 1.')
    if len(sample_ids) < 2:
        raise ValueError('At least two image-mask pairs are required.')

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(sample_ids), generator=generator).tolist()
    val_count = max(1, int(round(len(indices) * val_split)))
    val_count = min(val_count, len(indices) - 1)
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]
    train_ids = [sample_ids[index] for index in train_indices]
    val_ids = [sample_ids[index] for index in val_indices]
    return train_ids, val_ids


class SegmentationLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        ignore_index=None,
        class_weights=None,
        use_dice=False,
        dice_weight=0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.use_dice = use_dice
        self.dice_weight = dice_weight

        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        cross_entropy_options = {'weight': class_weights}
        if ignore_index is not None:
            cross_entropy_options['ignore_index'] = ignore_index
        self.cross_entropy = nn.CrossEntropyLoss(**cross_entropy_options)

    def forward(self, main_output, target):
        main_loss = self.cross_entropy(main_output, target)
        if self.use_dice:
            main_loss = main_loss + self.dice_weight * self._dice_loss(
                main_output,
                target,
            )

        return main_loss

    def _dice_loss(self, prediction, target):
        probabilities = torch.softmax(prediction, dim=1)
        valid_mask = torch.ones_like(target, dtype=torch.bool)
        safe_target = target.clone()

        if self.ignore_index is not None:
            valid_mask = target != self.ignore_index
            safe_target = safe_target.masked_fill(~valid_mask, 0)

        one_hot = F.one_hot(
            safe_target,
            num_classes=self.num_classes,
        ).permute(0, 3, 1, 2).to(probabilities.dtype)
        valid_mask = valid_mask.unsqueeze(1).to(probabilities.dtype)
        probabilities = probabilities * valid_mask
        one_hot = one_hot * valid_mask

        reduce_axes = (0, 2, 3)
        intersection = (probabilities * one_hot).sum(dim=reduce_axes)
        denominator = probabilities.sum(dim=reduce_axes) + one_hot.sum(dim=reduce_axes)
        dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
        return 1.0 - dice.mean()


def update_confusion_matrix(matrix, prediction, target, num_classes, ignore_index):
    prediction = prediction.reshape(-1)
    target = target.reshape(-1)
    valid = (target >= 0) & (target < num_classes)
    if ignore_index is not None:
        valid &= target != ignore_index
    indices = num_classes * target[valid] + prediction[valid]
    counts = torch.bincount(indices, minlength=num_classes ** 2)
    matrix += counts.reshape(num_classes, num_classes).cpu()


def metrics_from_confusion_matrix(matrix):
    matrix = matrix.to(torch.float64)
    true_positive = torch.diag(matrix)
    union = matrix.sum(dim=1) + matrix.sum(dim=0) - true_positive
    valid_classes = union > 0
    class_iou = torch.zeros_like(union)
    class_iou[valid_classes] = true_positive[valid_classes] / union[valid_classes]
    mean_iou = class_iou[valid_classes].mean().item() if valid_classes.any() else 0.0
    total = matrix.sum()
    pixel_accuracy = true_positive.sum().item() / total.item() if total > 0 else 0.0
    return mean_iou, pixel_accuracy


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    amp_enabled,
    grad_accum=1,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    progress = tqdm(loader, desc='Training')

    for step, (images, masks) in enumerate(progress, start=1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with autocast(enabled=amp_enabled):
            outputs = model(images)
            loss = criterion(outputs, masks)
            scaled_loss = loss / grad_accum

        scaler.scale(scaled_loss).backward()
        should_step = step % grad_accum == 0 or step == len(loader)
        if should_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        progress.set_postfix(loss=f'{loss.item():.4f}')

    return total_loss / max(1, len(loader))


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes, ignore_index, amp_enabled):
    model.eval()
    total_loss = 0.0
    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    for images, masks in tqdm(loader, desc='Validating'):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with autocast(enabled=amp_enabled):
            main_output = model(images)
            loss = criterion(main_output, masks)

        total_loss += loss.item()
        predictions = torch.argmax(main_output, dim=1)
        update_confusion_matrix(
            confusion_matrix,
            predictions,
            masks,
            num_classes,
            ignore_index,
        )

    mean_iou, pixel_accuracy = metrics_from_confusion_matrix(confusion_matrix)
    return total_loss / max(1, len(loader)), mean_iou, pixel_accuracy


def build_datasets(args):
    input_size = (args.input_height, args.input_width)
    discovery_dataset = SegmentationDataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        image_suffix=args.image_suffix,
        mask_suffix=args.mask_suffix,
        ignore_index=args.ignore_index,
        augment=False,
        input_size=input_size,
        binary_mask=args.num_classes == 2,
    )
    train_ids, val_ids = split_sample_ids(
        discovery_dataset.sample_ids,
        args.val_split,
        args.seed,
    )

    common_options = dict(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        image_suffix=args.image_suffix,
        mask_suffix=args.mask_suffix,
        ignore_index=args.ignore_index,
        input_size=input_size,
        binary_mask=args.num_classes == 2,
    )
    train_dataset = SegmentationDataset(
        **common_options,
        sample_ids=train_ids,
        augment=True,
    )
    val_dataset = SegmentationDataset(
        **common_options,
        sample_ids=val_ids,
        augment=False,
    )
    return train_dataset, val_dataset


def main(args):
    if args.grad_accum < 1:
        raise ValueError('grad_accum must be at least 1.')

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp_enabled = bool(args.amp and device.type == 'cuda')
    print(f'Device: {device}')
    print(f'Automatic mixed precision: {amp_enabled}')

    train_dataset, val_dataset = build_datasets(args)
    print(f'Training samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')

    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    if len(train_loader) == 0:
        raise RuntimeError('The training loader is empty. Reduce batch_size.')

    model = DeepLabV3Plus(
        num_classes=args.num_classes,
        backbone=args.backbone,
        output_stride=args.output_stride,
        pretrained=args.pretrained,
    ).to(device)
    criterion = SegmentationLoss(
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
        use_dice=args.use_dice,
        dice_weight=args.dice_weight,
    )
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.PolynomialLR(
        optimizer,
        total_iters=args.epochs,
        power=0.9,
    )
    scaler = GradScaler(enabled=amp_enabled)

    os.makedirs(args.save_dir, exist_ok=True)
    with open(
        os.path.join(args.save_dir, 'config.json'),
        'w',
        encoding='utf-8',
    ) as config_file:
        json.dump(vars(args), config_file, indent=2)

    best_mean_iou = -1.0
    for epoch in range(1, args.epochs + 1):
        print(f'\nEpoch {epoch}/{args.epochs}')
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            amp_enabled,
            args.grad_accum,
        )
        val_loss, val_mean_iou, val_accuracy = validate(
            model,
            val_loader,
            criterion,
            device,
            args.num_classes,
            args.ignore_index,
            amp_enabled,
        )
        scheduler.step()

        print(
            f'Train loss: {train_loss:.4f} | '
            f'Validation loss: {val_loss:.4f} | '
            f'Validation mIoU: {val_mean_iou:.4f} | '
            f'Validation accuracy: {val_accuracy:.4f}'
        )

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_miou': val_mean_iou,
            'config': vars(args),
        }
        if val_mean_iou > best_mean_iou:
            best_mean_iou = val_mean_iou
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_model.pth'))
            print(f'Saved best model with mIoU {best_mean_iou:.4f}.')

        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save(
                checkpoint,
                os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth'),
            )

    print(f'Training complete. Best validation mIoU: {best_mean_iou:.4f}')


if __name__ == '__main__':
    main(parse_args())
