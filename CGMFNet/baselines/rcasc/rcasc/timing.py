import csv
from pathlib import Path
from datetime import datetime, timezone, timedelta
import statistics
import torch
from .utils import dump

STAGES = ('prepare_ms', 'transfer_ms', 'forward_ms', 'postprocess_ms',
          'save_ms', 'detection_ms', 'total_ms')
SCOPES = {'prepare_ms': 'Image reading and preprocessing, including dataset construction', 'transfer_ms': 'Input tensor transfer to the compute device', 'forward_ms': 'Model forward pass with input on the compute device', 'postprocess_ms': 'Decoding, output transfer to CPU, and line fitting', 'save_ms': 'Overlay rendering and image/JSON saving, including image rereading', 'detection_ms': 'Image reading through detection coordinates, excluding rendering and saving', 'total_ms': 'Image reading through overlay and coordinate saving'}


def synchronize(device):
    dev = torch.device(device)
    if dev.type == 'cuda':
        torch.cuda.synchronize(dev)


def summarize(records):
    if not records:
        raise ValueError('Cannot summarize an empty timing run')
    result = []
    for stage in STAGES:
        values = [float(r[stage]) for r in records]
        total = sum(values)
        if total <= 0 or any(v < 0 for v in values):
            raise ValueError(f'Invalid duration for {stage}')
        result.append(dict(stage=stage, scope=SCOPES[stage], images=len(values),
                           total_ms=total, mean_ms=statistics.mean(values),
                           median_ms=statistics.median(values), min_ms=min(values),
                           max_ms=max(values), fps=len(values)*1000/total))
    return result


def write_reports(records, output, metadata):
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    folder = Path(output) / 'timing' / stamp
    folder.mkdir(parents=True, exist_ok=False)
    headers = {'index': 'index', 'image': 'image', 'width': 'width', 'height': 'height', 'rows': 'rows', 'prepare_ms': 'prepare_ms', 'transfer_ms': 'transfer_ms', 'forward_ms': 'forward_ms', 'postprocess_ms': 'postprocess_ms', 'save_ms': 'save_ms', 'detection_ms': 'detection_ms', 'total_ms': 'total_ms', 'forward_fps': 'forward_fps', 'detection_fps': 'detection_fps', 'total_fps': 'total_fps'}
    with (folder/'per_image_timing.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers.values())
        for record in records:
            writer.writerow([record[k] for k in headers])
    summary = summarize(records)
    with (folder/'fps_summary.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['stage', 'scope', 'images', 'total_ms', 'mean_ms', 'median_ms', 'min_ms', 'max_ms', 'fps'])
        for row in summary:
            writer.writerow(list(row.values()))
    metadata['timestamp_Asia_Shanghai'] = stamp
    metadata['summary'] = summary
    metadata['per_image'] = records
    dump(metadata, folder/'timing_metadata.json')
    return folder, summary
