# RCASC Rice-Row Detection Baseline

A local RCASC adaptation for RGB rice-row detection with a ResNet-18 backbone. Depth images are not required.

## Usage

Place images and LabelMe annotations using `line` or `linestrip` shapes in `inputs/`. Edit `configs/rice_all.json` for training settings and dataset splits. Each split group is the first 11 characters of an annotation filename stem; set `val_groups` and `test_groups` to match your data, leaving other groups for training.

Run from this directory:

```bash
pip install -r requirements.txt
python prepare_data.py --config configs/rice_all.json
python train.py
python evaluate.py --checkpoint runs/full/best.pt
python predict.py
```

The preparation command saves the selected configuration under `prepared/`, which training then uses. The supplied rice configuration uses pretrained ImageNet weights and 100 training epochs. Set `pretrained` to `false` for random initialization.

Checkpoints are saved under `runs/full/`, evaluation outputs under `evaluation/`, and predictions with timing reports under `predictions/`. For custom prediction paths, use `--checkpoint`, `--source`, and `--output`.

Prediction fits and extends straight lines to image boundaries by default. Use `--geometry polyline` to retain decoded anchor polylines. Evaluation uses decoded polylines without this straight-line postprocessing. Timing reports separate forward, detection, and total time including output saving.

This local adaptation does not establish equivalence to the original paper's results.
