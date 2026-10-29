import argparse
import time
import platform
from pathlib import Path
import torch
from PIL import Image
from rcasc.data import Rows
from rcasc.evaluation import decode, overlay
from rcasc.postprocess import fit_straight_rows
from rcasc.runtime import load_model
from rcasc.utils import dump
from rcasc.timing import synchronize, write_reports


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_DIR / 'runs' / 'full' / 'best.pt'
DEFAULT_SOURCE = PROJECT_DIR / 'inputs'
DEFAULT_OUTPUT = PROJECT_DIR / 'predictions'


def main():
    parser = argparse.ArgumentParser(description='Predict crop rows from images.')
    parser.add_argument('--checkpoint', default=str(DEFAULT_CHECKPOINT), help='Model checkpoint; defaults to runs/full/best.pt beside this script')
    parser.add_argument('--source', default=str(DEFAULT_SOURCE), help='Image file or directory; defaults to inputs beside this script')
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT), help='Prediction output directory')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--geometry', choices=['straight', 'polyline'], default='straight', help='Fit one straight segment per crop row, or retain raw anchor points')
    parser.add_argument('--warmup', type=int, default=20, help='Unrecorded warmup iterations before timing (default: 20)')
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error('--warmup must be non-negative')
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f'Checkpoint not found: {checkpoint}. Set DEFAULT_CHECKPOINT or pass --checkpoint.')
    source = Path(args.source).expanduser().resolve()
    print(f'Checkpoint: {checkpoint}', flush=True)
    print(f'Output: {Path(args.output).expanduser().resolve()}', flush=True)
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    images = sorted((p for p in source.iterdir() if p.suffix.lower() in extensions)) if source.is_dir() else [source]
    if not images or any((not p.is_file() or p.suffix.lower() not in extensions for p in images)):
        parser.error('No valid input images')
    load_start = time.perf_counter()
    model, config, device = load_model(checkpoint, args.device)
    synchronize(device)
    model_load_ms = (time.perf_counter() - load_start) * 1000
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    warmup_start = time.perf_counter()
    with torch.no_grad():
        if args.warmup:
            first = dict(image=str(images[0]), lines=[])
            warm_tensor, _, _ = Rows([first], config)[0]
            warm_tensor = warm_tensor[None].to(device)
            for _ in range(args.warmup):
                warm_lines = decode(model(warm_tensor), config)[0]
                if args.geometry == 'straight':
                    fit_straight_rows(warm_lines)
            del warm_tensor, warm_lines
        synchronize(device)
        warmup_ms = (time.perf_counter() - warmup_start) * 1000
        timing_records = []
        for index, image in enumerate(images, 1):
            synchronize(device)
            start = time.perf_counter()
            with Image.open(image) as im:
                width, height = im.size
            record = dict(image=str(image), width=width, height=height, lines=[])
            tensor, _, _ = Rows([record], config)[0]
            prepared = time.perf_counter()
            gpu_tensor = tensor[None].to(device)
            synchronize(device)
            transferred = time.perf_counter()
            logits = model(gpu_tensor)
            synchronize(device)
            forwarded = time.perf_counter()
            lines = decode(logits, config)[0]
            if args.geometry == 'straight':
                lines = fit_straight_rows(lines)
            synchronize(device)
            detected = time.perf_counter()
            overlay(record, lines, output / (image.name + '.overlay.jpg'), show_legend=False)
            dump(dict(image=str(image), width=width, height=height, normalized_coordinates=True,
                      geometry=args.geometry, lines=lines), output / (image.name + '.json'))
            saved = time.perf_counter()
            durations = dict(
                prepare_ms=(prepared-start)*1000,
                transfer_ms=(transferred-prepared)*1000,
                forward_ms=(forwarded-transferred)*1000,
                postprocess_ms=(detected-forwarded)*1000,
                save_ms=(saved-detected)*1000,
                detection_ms=(detected-start)*1000,
                total_ms=(saved-start)*1000,
            )
            timing_records.append(dict(index=index, image=image.name, width=width, height=height,
                rows=len(lines), **durations,
                forward_fps=1000/durations['forward_ms'],
                detection_fps=1000/durations['detection_ms'],
                total_fps=1000/durations['total_ms']))
            del logits, gpu_tensor
    metadata = dict(
        checkpoint=str(checkpoint), source=str(source), device=str(device),
        device_name=torch.cuda.get_device_name(torch.device(device)) if torch.device(device).type == 'cuda' else platform.processor(),
        torch_version=str(torch.__version__), python_version=platform.python_version(),
        input_height=config['height'], input_width=config['width'], batch_size=1,
        dtype=str(next(model.parameters()).dtype), cpu_threads=torch.get_num_threads(),
        geometry=args.geometry, warmup_iterations=args.warmup, warmup_ms=warmup_ms,
        model_load_ms=model_load_ms, images=len(images),
        fps_formula='image_count * 1000 / sum(duration_ms)',
        measurement='perf_counter wall clock with explicit CUDA synchronization at stage boundaries',
        exclusions='Model loading, warmup, directory scanning, terminal printing, timing report writing and per-record bookkeeping excluded from per-image times. File saves include OS caching, not disk fsync.',
        caveat='Synchronous batch=1 file pipeline. Not asynchronous video throughput. Small image counts and filesystem cache can affect results.',
    )
    report_dir, summary = write_reports(timing_records, output, metadata)
    print(f'Timing tables: {report_dir}')
    for row in summary:
        if row['stage'] in ('forward_ms', 'detection_ms', 'total_ms'):
            print(f"{row['stage']}: mean={row['mean_ms']:.3f} ms, FPS={row['fps']:.2f}")
if __name__ == '__main__':
    main()
