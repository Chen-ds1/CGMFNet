import argparse
import csv
import os
import time
from glob import glob

import cv2
import numpy as np
import torch
import yaml

import archs


IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp')


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate CGMFNet')
    parser.add_argument('--name', default='CGMFNet-L4', type=str)
    parser.add_argument('--model_dir', default='./models', type=str)
    parser.add_argument('--rgb_dir', default='./inputs/SeedingImg/images', type=str)
    parser.add_argument('--depth_dir', default='./inputs/SeedingImg/depths', type=str)
    parser.add_argument(
        '--mask_dir',
        default='./inputs/SeedingImg/rgb_masks/0',
        type=str,
    )
    parser.add_argument(
        '--csv_out',
        default='./evaluate_data/eval_results.csv',
        type=str,
    )
    parser.add_argument('--vis_dir', default='./vis_results', type=str)
    parser.add_argument('--threshold', default=None, type=float)
    return parser.parse_args()


def find_matching_file(directory, stem):
    for extension in IMAGE_EXTENSIONS:
        candidate = os.path.join(directory, stem + extension)
        if os.path.exists(candidate):
            return candidate
    return None


def preprocess_pair(rgb_img, depth_img, height, width, device):
    rgb_img = cv2.resize(
        rgb_img,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    depth_img = cv2.resize(
        depth_img,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    rgb_img = rgb_img.astype(np.float32) / 255.0
    depth_img = depth_img.astype(np.float32) / 255.0
    rgb_tensor = torch.from_numpy(
        rgb_img.transpose(2, 0, 1)
    ).unsqueeze(0).to(device)
    depth_tensor = torch.from_numpy(
        depth_img.transpose(2, 0, 1)
    ).unsqueeze(0).to(device)
    return rgb_tensor, depth_tensor


def compute_iou(prediction, target):
    intersection = np.logical_and(prediction, target).sum()
    union = np.logical_or(prediction, target).sum()
    return intersection / union if union > 0 else float('nan')


def load_model(args, device):
    config_path = os.path.join(args.model_dir, args.name, 'config.yml')
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f'Configuration not found: {config_path}')
    with open(config_path, 'r', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)

    model_path = os.path.join(args.model_dir, config['name'], 'model.pth')
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f'Model checkpoint not found: {model_path}')
    model = archs.__dict__[config['arch']](
        config['num_classes'],
        config['input_channels'],
        config['deep_supervision'],
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, config


def main(args):
    if not os.path.isdir(args.rgb_dir):
        raise FileNotFoundError(f'RGB directory not found: {args.rgb_dir}')
    if not os.path.isdir(args.depth_dir):
        raise FileNotFoundError(f'Depth directory not found: {args.depth_dir}')
    if not os.path.isdir(args.mask_dir):
        raise FileNotFoundError(f'Mask directory not found: {args.mask_dir}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, config = load_model(args, device)
    threshold = (
        args.threshold
        if args.threshold is not None
        else float(config.get('iou_threshold', 0.5))
    )
    print(f'Device: {device}')

    if args.vis_dir:
        os.makedirs(args.vis_dir, exist_ok=True)
    csv_parent = os.path.dirname(os.path.abspath(args.csv_out))
    os.makedirs(csv_parent, exist_ok=True)

    rgb_files = sorted(
        path
        for extension in IMAGE_EXTENSIONS
        for path in glob(os.path.join(args.rgb_dir, '*' + extension))
    )
    if not rgb_files:
        raise FileNotFoundError(f'No RGB images found in {args.rgb_dir}')

    results = []
    for index, rgb_path in enumerate(rgb_files, start=1):
        filename = os.path.basename(rgb_path)
        stem = os.path.splitext(filename)[0]
        depth_path = find_matching_file(args.depth_dir, stem)
        mask_path = find_matching_file(args.mask_dir, stem)
        if depth_path is None or mask_path is None:
            print(f'Skip {filename}: matching depth or mask is missing.')
            continue

        rgb_img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        depth_img = cv2.imread(depth_path, cv2.IMREAD_COLOR)
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if rgb_img is None or depth_img is None or mask_img is None:
            print(f'Skip {filename}: an image could not be decoded.')
            continue

        original_height, original_width = rgb_img.shape[:2]
        if mask_img.shape != (original_height, original_width):
            mask_img = cv2.resize(
                mask_img,
                (original_width, original_height),
                interpolation=cv2.INTER_NEAREST,
            )
        rgb_tensor, depth_tensor = preprocess_pair(
            rgb_img,
            depth_img,
            config['input_h'],
            config['input_w'],
            device,
        )

        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            output = model(rgb_tensor, depth_tensor)
            if isinstance(output, (list, tuple)):
                output = output[-1]
            probability = torch.sigmoid(output)[0, 0].cpu().numpy()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        probability = cv2.resize(
            probability,
            (original_width, original_height),
            interpolation=cv2.INTER_LINEAR,
        )
        prediction = probability > threshold
        target = mask_img > 127
        iou = compute_iou(prediction, target)
        fps = 1.0 / elapsed if elapsed > 0 else 0.0
        foreground_ratio = float(prediction.mean())
        results.append([filename, iou, elapsed, fps, foreground_ratio])

        if args.vis_dir:
            cv2.imwrite(
                os.path.join(args.vis_dir, f'{stem}_pred.png'),
                prediction.astype(np.uint8) * 255,
            )
            overlay = np.zeros(
                (original_height, original_width, 3),
                dtype=np.uint8,
            )
            overlay[target & prediction] = [0, 255, 0]
            overlay[~target & prediction] = [0, 0, 255]
            overlay[target & ~prediction] = [255, 0, 0]
            cv2.imwrite(
                os.path.join(args.vis_dir, f'{stem}_overlay.png'),
                overlay,
            )

        iou_text = 'N/A' if np.isnan(iou) else f'{iou:.4f}'
        print(
            f'[{index}/{len(rgb_files)}] {filename}: '
            f'IoU={iou_text}, foreground={foreground_ratio:.4f}, FPS={fps:.2f}'
        )

    with open(args.csv_out, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                'filename',
                'iou',
                'processing_time',
                'fps',
                'predicted_foreground_ratio',
            ]
        )
        for row in results:
            writer.writerow(
                [row[0], '' if np.isnan(row[1]) else row[1], *row[2:]]
            )

    if results:
        valid_ious = [row[1] for row in results if not np.isnan(row[1])]
        timing_rows = results[1:] if len(results) > 1 else results
        fps_values = np.asarray([row[3] for row in timing_rows])
        print(f'Threshold={threshold:.2f}, images={len(results)}')
        if valid_ious:
            print(
                f'Mean IoU: {np.mean(valid_ious):.4f} +/- '
                f'{np.std(valid_ious):.4f}'
            )
        print(
            f'Mean FPS: {fps_values.mean():.2f} +/- '
            f'{fps_values.std():.2f}'
        )
        print(f'CSV: {args.csv_out}')
        if args.vis_dir:
            print(f'Visualizations: {args.vis_dir}')


if __name__ == '__main__':
    main(parse_args())
