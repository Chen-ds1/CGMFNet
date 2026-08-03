import argparse
import os
from glob import glob

import albumentations as A
import cv2
import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import archs
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter, str2bool


def parse_args():
    parser = argparse.ArgumentParser(description='Validate CGMFNet')
    parser.add_argument('--name', default='CGMFNet-L4', type=str)
    parser.add_argument('--model_dir', default='./models', type=str)
    parser.add_argument('--plot', default=False, type=str2bool)
    parser.add_argument('--threshold', default=None, type=float)
    parser.add_argument('--output_dir', default='./outputs', type=str)
    parser.add_argument('--num_examples', default=3, type=int)
    return parser.parse_args()


def load_config(args):
    config_path = os.path.join(args.model_dir, args.name, 'config.yml')
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f'Configuration not found: {config_path}')
    with open(config_path, 'r', encoding='utf-8') as config_file:
        return yaml.safe_load(config_file)


def build_validation_dataset(config):
    image_pattern = os.path.join(
        'inputs',
        config['dataset'],
        config.get('rgb_folder', 'images'),
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
    _, validation_ids = train_test_split(
        image_ids,
        test_size=config.get('val_split', 0.2),
        random_state=config.get('seed', 41),
    )

    transform = A.Compose(
        [A.Resize(config['input_h'], config['input_w'])],
        additional_targets={
            'depth_image': 'image',
            'depth_mask': 'mask',
        },
    )
    return Dataset(
        img_ids=validation_ids,
        rgb_dir=os.path.join(
            'inputs', config['dataset'], config.get('rgb_folder', 'images')
        ),
        depth_dir=os.path.join(
            'inputs', config['dataset'], config.get('depth_folder', 'depths')
        ),
        rgb_mask_dir=os.path.join(
            'inputs',
            config['dataset'],
            config.get('rgb_mask_folder', 'rgb_masks'),
        ),
        depth_mask_dir=os.path.join(
            'inputs',
            config['dataset'],
            config.get('depth_mask_folder', 'depth_masks'),
        ),
        ext=config['ext'],
        num_classes=config['num_classes'],
        transform=transform,
    )


def main(args):
    config = load_config(args)
    threshold = (
        args.threshold
        if args.threshold is not None
        else float(config.get('iou_threshold', 0.5))
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    for key, value in config.items():
        print(f'{key}: {value}')

    model = archs.__dict__[config['arch']](
        config['num_classes'],
        config['input_channels'],
        config['deep_supervision'],
    ).to(device)
    model_path = os.path.join(args.model_dir, config['name'], 'model.pth')
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f'Model checkpoint not found: {model_path}')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    validation_dataset = build_validation_dataset(config)
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=device.type == 'cuda',
        drop_last=False,
    )

    for class_index in range(config['num_classes']):
        os.makedirs(
            os.path.join(args.output_dir, config['name'], str(class_index)),
            exist_ok=True,
        )

    meter = AverageMeter()
    last_batch = None
    with torch.inference_mode():
        for rgb_input, rgb_target, depth_input, depth_target, meta in tqdm(
            validation_loader,
            desc='Validating',
        ):
            rgb_input = rgb_input.to(device, non_blocking=True)
            depth_input = depth_input.to(device, non_blocking=True)
            rgb_target = rgb_target.to(device, non_blocking=True)
            depth_target = depth_target.to(device, non_blocking=True)

            output = model(rgb_input, depth_input)
            if isinstance(output, (list, tuple)):
                output = output[-1]

            iou = iou_score(output, rgb_target, threshold=threshold)
            meter.update(iou, rgb_input.size(0))
            probabilities = torch.sigmoid(output).cpu().numpy()

            for batch_index, sample_id in enumerate(meta['img_id']):
                for class_index in range(config['num_classes']):
                    output_path = os.path.join(
                        args.output_dir,
                        config['name'],
                        str(class_index),
                        sample_id + '.png',
                    )
                    cv2.imwrite(
                        output_path,
                        (probabilities[batch_index, class_index] * 255).astype(
                            np.uint8
                        ),
                    )

            last_batch = (
                rgb_input,
                rgb_target,
                depth_input,
                depth_target,
                output,
            )

    print(f'IoU: {meter.avg:.4f} (threshold={threshold:.2f})')
    if args.plot and last_batch is not None:
        plot_examples(
            *last_batch,
            threshold=threshold,
            num_examples=args.num_examples,
        )


def plot_examples(
    rgb_input,
    rgb_target,
    depth_input,
    depth_target,
    output,
    threshold=0.5,
    num_examples=3,
):
    import matplotlib.pyplot as plt

    del depth_target
    num_examples = min(num_examples, rgb_input.size(0))
    probabilities = torch.sigmoid(output).detach().cpu().numpy()
    figure, axes = plt.subplots(
        nrows=num_examples,
        ncols=4,
        figsize=(18, 4 * num_examples),
        squeeze=False,
    )

    for row in range(num_examples):
        rgb = np.transpose(rgb_input[row].cpu().numpy(), (1, 2, 0))
        depth = np.transpose(depth_input[row].cpu().numpy(), (1, 2, 0))
        target = rgb_target[row, 0].cpu().numpy()
        axes[row, 0].imshow(np.clip(rgb[:, :, ::-1], 0, 1))
        axes[row, 0].set_title('RGB branch input')
        axes[row, 1].imshow(depth[:, :, 0], cmap='gray')
        axes[row, 1].set_title('Depth branch input')
        axes[row, 2].imshow(probabilities[row, 0] > threshold, cmap='gray')
        axes[row, 2].set_title('Prediction')
        axes[row, 3].imshow(target, cmap='gray')
        axes[row, 3].set_title('Target')
        for axis in axes[row]:
            axis.axis('off')

    figure.tight_layout()
    plt.show()


if __name__ == '__main__':
    main(parse_args())
