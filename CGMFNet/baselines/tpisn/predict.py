import argparse
import csv
import json
import time
from pathlib import Path
from tpisn.paths import INPUT_DIR, PROJECT_DIR
import numpy as np
import torch
from tpisn.data import read_image, save_image
from tpisn.inference import load_model, predict, original_lines, overlay


def main():
    p = argparse.ArgumentParser(description='Infer masks, row instances and x=k*y+b center lines')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--input', default=str(INPUT_DIR), help='Image or image directory')
    p.add_argument('--output', default=str(PROJECT_DIR / 'predictions'))
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--eps', type=float, default=1.0)
    p.add_argument('--min-samples', type=int, default=200)
    p.add_argument('--cluster-method', choices=['fast', 'exact'], default='fast', help='fast: approximate anchor clustering; exact: DBSCAN')
    p.add_argument('--max-anchors', type=int, default=1024)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--warmup', type=int, default=10, help='Untimed full-pipeline warmup iterations')
    p.add_argument('--timing-csv', help='Default: OUTPUT/prediction_times.csv')
    args = p.parse_args()
    if args.warmup < 0:
        p.error('--warmup must be nonnegative')
    source, dest = Path(args.input), Path(args.output)
    paths = sorted(x for x in source.rglob('*') if x.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp')) if source.is_dir() else [source]
    if not paths:
        p.error('No input images found')
    model = load_model(args.checkpoint, args.device)
    dest.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    def detect(image):
        probability, instances, lines = predict(model, image, device, args.threshold, args.eps,
                                                args.min_samples, args.cluster_method, args.max_anchors)
        return probability, instances, original_lines(lines, image.shape[1], image.shape[0])

    if args.warmup:
        warmup_image = read_image(paths[0])
        for _ in range(args.warmup):
            detect(warmup_image)
        sync()
        del warmup_image
    csv_path = Path(args.timing_csv) if args.timing_csv else dest / 'prediction_times.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    columns = ['index', 'image_name', 'image_path', 'image_width', 'image_height', 'row_count',
               'read_ms', 'detection_ms', 'detection_fps', 'render_save_ms',
               'total_ms', 'cluster_method', 'max_anchors']

    with csv_path.open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for index, path in enumerate(paths):
            start = time.perf_counter()
            image = read_image(path)
            read_end = time.perf_counter()
            sync()
            detect_start = time.perf_counter()
            probability, instances, lines = detect(image)
            sync()
            detect_end = time.perf_counter()
            name = '{:06d}_{}'.format(index, path.stem)
            save_image(dest / (name + '_overlay.png'), overlay(image, instances, lines))
            save_image(dest / (name + '_mask.png'), (probability >= args.threshold).astype(np.uint8) * 255)
            save_image(dest / (name + '_instances.png'), instances.astype(np.uint16))
            (dest / (name + '_lines.json')).write_text(json.dumps(dict(source=str(path),
                image_size=[image.shape[1], image.shape[0]], mask_size=[512, 256],
                line_coordinates='original image; x=k*y+b', cluster_method=args.cluster_method,
                max_anchors=args.max_anchors, lines=lines), indent=2), encoding='utf-8')
            end = time.perf_counter()
            detection_seconds = detect_end - detect_start
            record = dict(zip(columns, [index+1, path.name, str(path.resolve()), image.shape[1], image.shape[0],
                len(lines), (read_end-start)*1000, detection_seconds*1000, 1/detection_seconds,
                (end-detect_end)*1000, (end-start)*1000, args.cluster_method, args.max_anchors]))
            writer.writerow(record)
            file.flush()
            records.append(record)
    times = np.array([r['detection_ms'] for r in records])
    summary = dict(images=len(records), detection_mean_ms=float(times.mean()),
                   detection_p95_ms=float(np.percentile(times,95)),
                   detection_overall_fps=float(1000*len(records)/times.sum()),
                   per_image_total_mean_ms=float(np.mean([r['total_ms'] for r in records])),
                   checkpoint=str(Path(args.checkpoint).resolve()), args=vars(args),
                   gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
                   torch_version=torch.__version__, csv=str(csv_path.resolve()),
                   timing_scope='Detection includes preprocessing, device transfers, network, clustering, WLS, coordinate conversion. CUDA synchronized.',
                   excluded_from_detection='Model loading, warmup, disk reading, drawing, saving, CSV logging and printing.',
                   total_scope='Per-image total includes reading, synchronization, detection, drawing and output saving; excludes CSV logging/printing, model loading and warmup.')
    summary_path = csv_path.with_name(csv_path.stem + '_summary.json')
    summary_path.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Timing CSV:', csv_path.resolve())
    print('Overall detection FPS: {:.2f}'.format(summary['detection_overall_fps']))


if __name__ == '__main__':
    main()
