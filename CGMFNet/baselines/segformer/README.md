# SegFormer Segmentation Baseline

This directory contains a reproducible SegFormer baseline for semantic
segmentation. It supports MiT-B0 through MiT-B5 backbones, binary or multiclass
masks, optional Dice loss, mixed-precision training, batch evaluation, and file
or directory inference.

## Structure

```text
segformer/
|-- dataset.py
|-- train.py
|-- evaluate.py
|-- predict.py
|-- requirements.txt
`-- segformer/
    |-- __init__.py
    |-- mit.py
    `-- segformer.py
```

Place matching images and masks under `inputs/images` and `inputs/masks`.
Matching files must have the same stem. Binary masks may use either `0/1` or
`0/255` values.

## Installation

```bash
pip install -r requirements.txt
```

## Training

Defaults are defined in `parse_args()`, so the script can be launched directly
from an IDE:

```bash
python train.py
```

Command-line arguments override those defaults:

```bash
python train.py --epochs 100 --batch_size 2 --backbone B0
```

Boolean values accept `true` or `false`, for example `--amp false`.

## Evaluation

```bash
python evaluate.py
```

The evaluator writes per-image IoU and inference speed to CSV. It can also save
prediction masks and error overlays.

## Prediction

```bash
python predict.py
```

The default input is `inputs/images/1.png`. The `--input` argument also accepts
a directory for batch prediction. Set both input dimensions to `0` to preserve
the original image resolution during inference.

## Pretrained weights

Automatic pretrained MiT weight download is not included. Load external
backbone weights explicitly if they are required, and document their source and
license in the main repository.

## Reproducibility

- Training and validation use separate dataset objects.
- The random seed controls dataset splitting and model initialization.
- Checkpoints include the training configuration.
- CUDA timing is synchronized during evaluation.
- Input dimensions are explicitly named height and width.
