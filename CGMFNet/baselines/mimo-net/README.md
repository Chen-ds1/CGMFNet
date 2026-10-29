# MIMO-Net Rice-Row Detection Baseline

A local MIMO-Net adaptation for plant segmentation and rice-row detection using RGB and three-channel pseudocolor depth images. The default configuration trains without pretrained weights.

## Usage

Edit paths and training settings in `config.py`. Training and evaluation require matching sample IDs in `rgb`, `depths`, `plant_masks`, and `row_lines` under `DATA_ROOT`. Row annotations use LabelMe `rice_row` lines. Keep `splits.json` consistent with your dataset and checkpoint.

```bash
pip install -r requirements.txt
python prepare_data.py
python train.py
python evaluate.py
python predict.py
```

The scripts also run directly in PyCharm. Set `CHECKPOINT` to the intended `best.pt` before evaluation or prediction. For unlabeled prediction, set `PREDICT_RGB` and `PREDICT_DEPTH` to folders containing paired PNG files with identical names; labels are not required.

Results are saved under `RUN_ROOT`, including masks, row coordinates, overlays, timing records, and an evaluation workbook. Choose a new `RUN_ROOT` for a new training experiment.

Keep `postprocess_config.json` to preserve the selected row-extraction settings. Optionally run `validate_postprocess.py` after training to select new settings on the validation split; this overwrites that file. Extended lines are exported for visualization, while row metrics use the extracted segments. Centimeter errors require validated ground homographies.

This local adaptation does not establish equivalence to the original paper's results.
