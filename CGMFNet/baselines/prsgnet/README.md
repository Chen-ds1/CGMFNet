# PRSGNet Rice-Row Detection Baseline

A local PRSGNet implementation for RGB rice-row detection. The default configuration trains without pretrained weights. Depth images are not required.

## Usage

Place images and LabelMe row annotations in `inputs/`. Set paths, annotation labels, and training options in `configs/default.json`, then run:

```bash
pip install -r requirements.txt
python train.py
python calibrate.py
python evaluate.py
python predict.py
```

Training automatically prepares missing dataset manifests and saves checkpoints under `runs/`. Calibration is optional and selects a confidence threshold using the validation split. Evaluation uses the test split; add `--split val` for validation. Prediction also accepts unlabeled images through `--input`.

By default, inference selects the most recently modified `runs/*/best.pt`. Set `inference.weights` or use `--weights` to select a specific checkpoint. The confidence threshold comes from an explicit `--score` or configuration value, then the checkpoint's `calibration.json`, otherwise 0.5.

Prediction and evaluation outputs are saved in timestamped folders under `results/`, including row coordinates, overlays, and timing records. Lines are extended to image boundaries for display by default; use `--line-mode raw` to disable this. Evaluation metrics always use raw predictions.

This local implementation does not establish equivalence to the original paper's results.
