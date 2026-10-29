from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment
from .data import Rows
from .utils import dump

def decode(out, c):
    p, e = out
    pos = ((p.softmax(-1) * torch.arange(c['bins'], device=p.device)).sum(-1) + 0.5) / c['bins']
    pos = pos.cpu().numpy()
    valid = (e.softmax(-1)[..., 1] > c['existence_threshold']).cpu().numpy()
    anchors = (np.arange(c['anchors']) + 0.5) / c['anchors']
    result = []
    for batch in range(len(pos)):
        lines = []
        for lane in range(c['max_rows']):
            axis = int(valid[batch, 1, lane].sum() > valid[batch, 0, lane].sum())
            idx = np.flatnonzero(valid[batch, axis, lane])
            segments = []
            current = []
            previous = -2
            for j in idx:
                if j != previous + 1 and current:
                    segments.append(current)
                    current = []
                xy = [float(pos[batch, axis, lane, j]), float(anchors[j])]
                current.append(xy if axis == 0 else xy[::-1])
                previous = j
            if current:
                segments.append(current)
            segments = [s for s in segments if len(s) >= c['min_points']]
            if segments:
                lines.append(dict(slot=lane + 1, branch='row' if axis == 0 else 'column', segments=segments))
        result.append(lines)
    return result

def mask_line(segments, w, h, width):
    im = Image.new('1', (w, h))
    d = ImageDraw.Draw(im)
    for p in segments:
        xy = [(round(x * (w - 1)), round(y * (h - 1))) for x, y in p]
        if len(xy) > 1:
            d.line(xy, fill=1, width=width)
    return np.asarray(im, bool)

def match_counts(pred, record, width):
    w, h = (record['width'], record['height'])
    pm = [mask_line(p['segments'], w, h, width) for p in pred]
    gm = [mask_line([p], w, h, width) for p in record['lines']]
    ious = np.zeros((len(pm), len(gm)))
    for i, p in enumerate(pm):
        for j, g in enumerate(gm):
            ious[i, j] = (p & g).sum() / max(1, (p | g).sum())
    tp = 0
    if ious.size:
        rows, cols = linear_sum_assignment(-((ious > 0.5) * 1000 + ious))
        tp = int((ious[rows, cols] > 0.5).sum())
    return (tp, len(pm) - tp, len(gm) - tp)

def overlay(record, pred, path, show_legend=True):
    im = Image.open(record['image']).convert('RGB')
    d = ImageDraw.Draw(im)
    w, h = im.size
    for p in record.get('lines', []):
        d.line([(x * (w - 1), y * (h - 1)) for x, y in p], fill=(30, 220, 30), width=2)
    for line in pred:
        for p in line['segments']:
            d.line([(x * (w - 1), y * (h - 1)) for x, y in p], fill=(255, 40, 40), width=3)
    if show_legend:
        title = 'GT: green | prediction: red' if record.get('lines') else 'Prediction: red'
        d.rectangle((0, 0, 325, 22), fill='black')
        d.text((5, 5), title, fill='white')
    im.save(path)

@torch.no_grad()
def evaluate(model, records, c, device, out=None):
    model.eval()
    tp = fp = fn = 0
    details = []
    loader = DataLoader(Rows(records, c), batch_size=c['batch_size'], shuffle=False, num_workers=0)
    offset = 0
    for x, _, _ in loader:
        preds = decode(model(x.to(device)), c)
        for pred in preds:
            r = records[offset]
            offset += 1
            t, p, n = match_counts(pred, r, c['line_width'])
            tp += t
            fp += p
            fn += n
            details.append(dict(id=r['id'], tp=t, fp=p, fn=n, lines=pred))
            if out:
                Path(out).mkdir(parents=True, exist_ok=True)
                overlay(r, pred, Path(out) / (r['id'] + '.jpg'))
    metrics = dict(tp=tp, fp=fp, fn=fn, precision=tp / max(1, tp + fp), recall=tp / max(1, tp + fn), f1=2 * tp / max(1, 2 * tp + fp + fn), images=len(records), line_width=c['line_width'], iou_threshold=0.5)
    if out:
        dump(metrics, Path(out) / 'metrics.json')
        dump(details, Path(out) / 'predictions.json')
    return metrics
