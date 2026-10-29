from pathlib import Path
ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / 'inputs'
RUN_ROOT = ROOT / 'runs' / 'rice_v2'
SPLITS = ROOT / 'splits.json'
SEED = 42
IMAGE_HW = (480, 640)
BACKBONE = 'maxvit_tiny_tf_224'
WIDTH = 64
DEPTH_MODE = 'pseudocolor'
DEPTH_SCALE_METERS = 0.001
PRETRAINED = False
EPOCHS = 100
BATCH_SIZE = 2
LR = 0.0001
EARLY_STOP_PATIENCE = 15
NUM_WORKERS = 0
LINE_WIDTH = 5
WARMUP = 20
THRESHOLD = 0.5
MIN_COMPONENT_AREA = 25
MIN_ROW_SPAN = 0.15
MATCH_MAX_PIXEL_DISTANCE = 30.0
MATCH_MAX_ANGLE_DEG = 15.0
MIN_VERTICAL_OVERLAP = 0.5


GROUND_HOMOGRAPHIES = None
CHECKPOINT = RUN_ROOT / 'best.pt'
if not CHECKPOINT.exists():
    CHECKPOINT = ROOT / 'runs' / 'rice_v1' / 'best.pt'
PREDICT_RGB = ROOT / 'inputs' / 'rgb'
PREDICT_DEPTH = ROOT / 'inputs' / 'depths'
