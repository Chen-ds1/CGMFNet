from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_DIR / 'inputs'
PREPARED_DIR = PROJECT_DIR / 'prepared'

DEFAULT = dict(height=480, width=640, anchors=36, bins=100, max_rows=9, hidden=256, pool_channels=8, lambda_expect=1.0, delta_exist=1.0, lr=0.004, weight_decay=0.0001, batch_size=8, epochs=100, warmup=5, seed=42, mode='all', pretrained=True, existence_threshold=0.5, min_points=3, line_width=40)
