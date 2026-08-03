import argparse
import os

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from deeplabv3plus.deeplabv3plus import DeepLabV3Plus


IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


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
    parser = argparse.ArgumentParser(description='Run DeepLabV3+ inference')
    parser.add_argument('--input', default='./inputs/images/1.png', type=str)
    parser.add_argument('--output', default='./predictions', type=str)
    parser.add_argument('--checkpoint', default='./models/best_model.pth', type=str)
    parser.add_argument('--input_height', default=240, type=int)
    parser.add_argument('--input_width', default=320, type=int)
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument(
        '--backbone',
        default='resnet50',
        choices=['resnet34', 'resnet50', 'resnet101'],
    )
    parser.add_argument('--output_stride', default=16, type=int, choices=[8, 16])
    parser.add_argument('--binary', default=True, type=str2bool)
    parser.add_argument('--colorize', default=False, type=str2bool)
    return parser.parse_args()


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict'], checkpoint.get('config', {})
    return checkpoint, {}


def create_palette():
    palette = []
    for label in range(256):
        red = green = blue = 0
        value = label
        for bit in range(8):
            red |= ((value >> 0) & 1) << (7 - bit)
            green |= ((value >> 1) & 1) << (7 - bit)
            blue |= ((value >> 2) & 1) << (7 - bit)
            value >>= 3
        palette.extend([red, green, blue])
    return palette


@torch.inference_mode()
def predict_image(model, image_path, device, input_size):
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    if input_size is not None:
        image = TF.resize(
            image,
            input_size,
            interpolation=TF.InterpolationMode.BILINEAR,
        )

    tensor = TF.to_tensor(image)
    tensor = TF.normalize(
        tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ).unsqueeze(0).to(device)

    output = model(tensor)
    prediction = torch.argmax(output, dim=1)[0].cpu().numpy().astype(np.uint8)
    nearest_resampling = getattr(Image, 'Resampling', Image).NEAREST
    if prediction.shape[::-1] != original_size:
        prediction = np.asarray(
            Image.fromarray(prediction).resize(original_size, nearest_resampling)
        )
    return prediction


def save_prediction(prediction, output_path, num_classes, binary, colorize):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if binary and num_classes == 2:
        output = Image.fromarray((prediction > 0).astype(np.uint8) * 255)
    elif colorize:
        output = Image.fromarray(prediction, mode='P')
        output.putpalette(create_palette())
    else:
        output = Image.fromarray(prediction)
    output.save(output_path)


def output_path_for_file(input_path, output_path, batch_mode):
    filename = os.path.splitext(os.path.basename(input_path))[0] + '.png'
    if batch_mode:
        return os.path.join(output_path, filename)
    if os.path.splitext(output_path)[1].lower() in IMAGE_EXTENSIONS:
        return output_path
    return os.path.join(output_path, filename)


def main(args):
    if not os.path.exists(args.input):
        raise FileNotFoundError(f'Input path not found: {args.input}')
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f'Checkpoint not found: {args.checkpoint}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict, checkpoint_config = load_checkpoint(args.checkpoint, device)
    backbone = checkpoint_config.get('backbone', args.backbone)
    num_classes = int(checkpoint_config.get('num_classes', args.num_classes))
    output_stride = int(checkpoint_config.get('output_stride', args.output_stride))
    input_height = int(checkpoint_config.get('input_height', args.input_height))
    input_width = int(checkpoint_config.get('input_width', args.input_width))
    input_size = None
    if input_height > 0 and input_width > 0:
        input_size = (input_height, input_width)

    model = DeepLabV3Plus(
        num_classes=num_classes,
        backbone=backbone,
        output_stride=output_stride,
        pretrained=False,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    batch_mode = os.path.isdir(args.input)
    if batch_mode:
        image_paths = [
            os.path.join(args.input, name)
            for name in sorted(os.listdir(args.input))
            if name.lower().endswith(IMAGE_EXTENSIONS)
        ]
        if not image_paths:
            raise RuntimeError(f'No images were found in {args.input}.')
        os.makedirs(args.output, exist_ok=True)
    else:
        image_paths = [args.input]

    for index, image_path in enumerate(image_paths, start=1):
        prediction = predict_image(model, image_path, device, input_size)
        output_path = output_path_for_file(image_path, args.output, batch_mode)
        save_prediction(
            prediction,
            output_path,
            num_classes,
            args.binary,
            args.colorize,
        )
        print(f'[{index}/{len(image_paths)}] Saved: {output_path}')


if __name__ == '__main__':
    main(parse_args())
