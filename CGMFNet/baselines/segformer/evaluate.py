import argparse
import csv
import os
import time

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from segformer.segformer import SegFormer


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate SegFormer')
    parser.add_argument('--image_dir', default='./inputs/images', type=str)
    parser.add_argument('--mask_dir', default='./inputs/masks', type=str)
    parser.add_argument('--checkpoint', default='./models/best_model.pth', type=str)
    parser.add_argument('--csv_out', default='./eval_results.csv', type=str)
    parser.add_argument('--vis_dir', default='./vis_results', type=str)
    parser.add_argument('--input_height', default=240, type=int)
    parser.add_argument('--input_width', default=320, type=int)
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--class_id', default=1, type=int)
    parser.add_argument(
        '--backbone',
        default='B0',
        choices=['B0', 'B1', 'B2', 'B3', 'B4', 'B5'],
    )
    return parser.parse_args()


def find_mask(mask_dir, sample_id):
    for extension in ('.png', '.jpg', '.jpeg', '.bmp'):
        candidate = os.path.join(mask_dir, sample_id + extension)
        if os.path.exists(candidate):
            return candidate
    return None


def compute_iou(prediction, target, class_id=1):
    predicted_class = prediction == class_id
    target_class = target == class_id
    intersection = np.logical_and(predicted_class, target_class).sum()
    union = np.logical_or(predicted_class, target_class).sum()
    return intersection / union if union > 0 else float('nan')


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict'], checkpoint.get('config', {})
    return checkpoint, {}


def preprocess_image(image, input_size):
    resized = TF.resize(
        image,
        input_size,
        interpolation=TF.InterpolationMode.BILINEAR,
    )
    tensor = TF.to_tensor(resized)
    tensor = TF.normalize(
        tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    return tensor.unsqueeze(0)


def main(args):
    if not os.path.isdir(args.image_dir):
        raise FileNotFoundError(f'Image directory not found: {args.image_dir}')
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f'Checkpoint not found: {args.checkpoint}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict, checkpoint_config = load_checkpoint(args.checkpoint, device)

    backbone = checkpoint_config.get('backbone', args.backbone)
    num_classes = int(checkpoint_config.get('num_classes', args.num_classes))
    input_height = int(checkpoint_config.get('input_height', args.input_height))
    input_width = int(checkpoint_config.get('input_width', args.input_width))

    model = SegFormer(
        num_classes=num_classes,
        backbone=backbone,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = sorted(
        name
        for name in os.listdir(args.image_dir)
        if name.lower().endswith(extensions)
    )
    if not image_files:
        raise RuntimeError(f'No images were found in {args.image_dir}.')

    if args.vis_dir:
        os.makedirs(args.vis_dir, exist_ok=True)
    csv_parent = os.path.dirname(os.path.abspath(args.csv_out))
    os.makedirs(csv_parent, exist_ok=True)

    results = []
    input_size = (input_height, input_width)
    for index, filename in enumerate(image_files, start=1):
        image_path = os.path.join(args.image_dir, filename)
        sample_id = os.path.splitext(filename)[0]
        mask_path = find_mask(args.mask_dir, sample_id)

        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        tensor = preprocess_image(image, input_size).to(device)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        with torch.inference_mode():
            output = model(tensor)
            if isinstance(output, tuple):
                output = output[0]
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        prediction = torch.argmax(output, dim=1)[0].cpu().numpy().astype(np.uint8)
        nearest_resampling = getattr(Image, "Resampling", Image).NEAREST
        prediction = np.asarray(
            Image.fromarray(prediction).resize(original_size, nearest_resampling)
        )
        fps = 1.0 / elapsed if elapsed > 0 else 0.0

        iou = None
        target = None
        if mask_path is not None:
            target_image = Image.open(mask_path).convert('L')
            if target_image.size != original_size:
                target_image = target_image.resize(original_size, nearest_resampling)
            target = np.asarray(target_image)
            if num_classes == 2 and target.max(initial=0) > 1:
                target = (target > 127).astype(np.uint8)
            iou = compute_iou(prediction, target, args.class_id)

        results.append([filename, iou, elapsed, fps])
        iou_text = 'N/A' if iou is None or np.isnan(iou) else f'{iou:.4f}'
        print(
            f'[{index}/{len(image_files)}] {filename}: '
            f'IoU={iou_text}, time={elapsed:.4f}s, FPS={fps:.2f}'
        )

        if args.vis_dir and target is not None:
            predicted_class = prediction == args.class_id
            target_class = target == args.class_id
            visualization = np.zeros((*target.shape, 3), dtype=np.uint8)
            visualization[target_class & predicted_class] = [0, 255, 0]
            visualization[~target_class & predicted_class] = [255, 0, 0]
            visualization[target_class & ~predicted_class] = [0, 0, 255]
            Image.fromarray(visualization).save(
                os.path.join(args.vis_dir, f'{sample_id}_overlay.png')
            )
            Image.fromarray((predicted_class * 255).astype(np.uint8)).save(
                os.path.join(args.vis_dir, f'{sample_id}_pred.png')
            )

    with open(args.csv_out, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['filename', 'iou', 'processing_time', 'fps'])
        for filename, iou, elapsed, fps in results:
            writer.writerow(
                [
                    filename,
                    '' if iou is None or np.isnan(iou) else iou,
                    elapsed,
                    fps,
                ]
            )

    valid_ious = [row[1] for row in results if row[1] is not None and not np.isnan(row[1])]
    timing_rows = results[1:] if len(results) > 1 else results
    fps_values = [row[3] for row in timing_rows]
    print(f'Evaluated images: {len(results)}')
    if valid_ious:
        print(f'Mean IoU: {np.mean(valid_ious):.4f} +/- {np.std(valid_ious):.4f}')
    print(f'Mean FPS: {np.mean(fps_values):.2f} +/- {np.std(fps_values):.2f}')
    print(f'Results saved to: {args.csv_out}')


if __name__ == '__main__':
    main(parse_args())
