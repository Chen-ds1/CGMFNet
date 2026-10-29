# TP-ISN Rice-Row Detection Baseline

A local TP-ISN implementation for RGB rice-row detection using segmentation, instance embeddings, clustering, and line fitting. The MobileNetV3-Small backbone is initialized without pretrained weights. Depth images are not required.

## Usage

Place RGB images and LabelMe line annotations in `inputs/`, then run from this directory:

```bash
pip install -r requirements.txt
python prepare_data.py
python train.py
python evaluate.py --checkpoint runs/tpisn/best.pt
python predict.py --checkpoint runs/tpisn/best.pt --input inputs
```

Annotations must use `line` or `linestrip` shapes. Data preparation creates `rice_data/` and requires an empty output directory. Skip this step if the prepared dataset already exists. Paths can be changed through command-line arguments; use `--help` for options.

Training checkpoints are saved under `runs/tpisn/`. Prediction masks, instances, row lines, overlays, and timing records are saved under `predictions/`. Evaluation results are saved to `evaluation.json`.

Prediction and evaluation default to `--cluster-method fast`, which uses approximate anchor clustering. Use `--cluster-method exact` for full DBSCAN, and keep the same mode when comparing accuracy and speed. Training recalibrates BatchNorm statistics using the training split before validation.

For optional speed measurements:

```bash
python benchmark_fps.py --checkpoint runs/tpisn/best.pt --data rice_data
```

The benchmark reports network and detection-pipeline timing separately. This local implementation does not establish equivalence to the original paper's results.
