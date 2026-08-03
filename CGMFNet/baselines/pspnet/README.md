# PSPNet Segmentation Baseline

This directory contains a reproducible PSPNet baseline for semantic segmentation.
It supports ResNet-34, ResNet-50, and ResNet-101 backbones, auxiliary supervision,
optional Dice loss, mixed-precision training, batch evaluation, and model-complexity
reporting.

## Directory structure

```text
pspnet/
|-- dataset.py
|-- train.py
|-- evaluate.py
|-- get_params.py
|-- requirements.txt
|-- pspnet/
|   |-- __init__.py
|   `-- pspnet.py
|-- inputs/
|   |-- images/
|   `-- masks/
|-- models/
`-- vis_results/
```

Images and masks must have matching file stems. For binary segmentation, masks may
contain either `0/1` or `0/255` pixel values.

## Installation

```bash
pip install -r requirements.txt
```

## Training

The defaults in `parse_args()` allow the file to run directly from an IDE:

```bash
python train.py
```

Command-line arguments override the source defaults:

```bash
python train.py --epochs 100 --batch_size 4 --backbone resnet34
```

Boolean arguments accept `true` or `false`:

```bash
python train.py --amp true --pretrained true --use_dice false
```

The best checkpoint is saved as `models/best_model.pth`. The checkpoint stores the
model state, optimizer state, scheduler state, validation mIoU, and training
configuration.

## Evaluation

```bash
python evaluate.py
```

Example with explicit paths:

```bash
python evaluate.py --checkpoint ./models/best_model.pth \
  --image_dir ./inputs/images --mask_dir ./inputs/masks
```

The script writes per-image IoU and inference speed to CSV. If `vis_dir` is not an
empty string, it also saves prediction masks and error overlays.

## Model complexity

```bash
python get_params.py --backbone resnet34 --input_height 240 --input_width 320
```

## Reproducibility notes

- Training and validation use separate dataset objects, so validation settings do
  not disable training augmentation.
- The random seed controls the dataset split and model initialization.
- CUDA timing is synchronized during evaluation.
- Input dimensions are explicitly named `input_height` and `input_width`.
- Use the same split, input size, optimizer, loss, and evaluation protocol when
  comparing PSPNet with other segmentation models.


