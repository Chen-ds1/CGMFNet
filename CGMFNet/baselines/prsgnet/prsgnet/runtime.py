import argparse
import json
from datetime import datetime
from pathlib import Path
import torch

PROJECT = Path(__file__).resolve().parents[1]


def resolve(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT/path).resolve()


def load_config(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', default='configs/default.json')
    args, _ = parser.parse_known_args(argv)
    path = resolve(args.config)
    config = json.loads(path.read_text(encoding='utf-8-sig'))
    return config, path


def device_name(value):
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if value == 'auto' else value
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Set --device cpu.')
    return device


def choose_weights(value=None):
    if value:
        path = resolve(value)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    candidates = list((PROJECT/'runs').glob('*/best.pt'))
    if not candidates:
        raise FileNotFoundError('No runs/*/best.pt found. Train first or specify --weights.')
    path = max(candidates, key=lambda p:(p.stat().st_mtime_ns,str(p)))
    print(f'Auto-selected newest best.pt: {path}', flush=True)
    return path


def choose_score(value, weights):
    if value is not None:
        score, source = float(value), 'explicit CLI/config'
    elif (weights.parent/'calibration.json').is_file():
        data = json.loads((weights.parent/'calibration.json').read_text(encoding='utf-8-sig'))
        score, source = float(data['selected']['score']), str(weights.parent/'calibration.json')
    else:
        score, source = .5, 'fallback 0.5 (no calibration.json found)'
    if not 0 <= score <= 1:
        raise ValueError('score must lie in [0,1]')
    print(f'Confidence threshold: {score:g}; source: {source}', flush=True)
    return score


def new_output(kind, value=None):
    path = resolve(value) if value else PROJECT/'results'/kind/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {path}. Choose a new --out.')
    path.mkdir(parents=True, exist_ok=True)
    return path


def add_inference_arguments(parser, config, config_path):
    inference = config['inference']
    parser.add_argument('--config', default=str(config_path))
    parser.add_argument('--weights', default=inference.get('weights'))
    parser.add_argument('--score', type=float, default=inference.get('score'))
    parser.add_argument('--nms-distance', type=float, default=inference.get('nms_distance',15))
    parser.add_argument('--line-mode', choices=['raw','extended'], default=inference.get('line_mode','extended'), help='Display/export geometry only; metrics always use raw predictions')
    parser.add_argument('--warmup', type=int, default=inference.get('warmup',50), help='Untimed warmup iterations before per-image timing')
    parser.add_argument('--device', default=config['train'].get('device','auto'))
    parser.add_argument('--out', help='New output folder; relative to project root')


def load_model(weights, device):
    from prsgnet.model import PRSGNet
    torch.set_num_threads(4)
    model = PRSGNet().to(device)
    checkpoint = torch.load(weights,map_location=device,weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model
