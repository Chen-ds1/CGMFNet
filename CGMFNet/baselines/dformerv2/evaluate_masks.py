import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def binary(path):
    with Image.open(path) as im:
        a = np.asarray(im)
    if a.ndim != 2:
        raise ValueError(f"Expected single-channel binary mask: {path}")
    if not set(np.unique(a).tolist()).issubset({0, 1, 255}):
        raise ValueError(f"Mask must contain only 0/1 or 0/255: {path}")
    return a > 0


def scores(tp, fp, fn, tn):
    def ratio(a, b):
        return 100.0 * a / b if b else None
    return dict(iou=ratio(tp, tp + fp + fn),
                precision=ratio(tp, tp + fp), recall=ratio(tp, tp + fn),
                dice=ratio(2 * tp, 2 * tp + fp + fn),
                pixel_accuracy=ratio(tp + tn, tp + fp + fn + tn))


def summary(rows):
    totals = {key: sum(r[key] for r in rows) for key in ('tp', 'fp', 'fn', 'tn')}
    result = {'n_images': len(rows), 'micro_percent': scores(**totals),
              'confusion_counts': totals, 'macro_percent': {}}
    for key in scores(1, 0, 0, 0):
        values = [r[key] for r in rows if r[key] is not None]
        result['macro_percent'][key] = {
            'n_defined': len(values), 'n_undefined': len(rows) - len(values),
            'mean': float(np.mean(values)) if values else None,
            'sd': float(np.std(values, ddof=1)) if len(values) > 1 else None}
    return result


def main():
    p = argparse.ArgumentParser(description='Evaluate binary segmentation masks on an explicit file list.')
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--split', type=Path, required=True,
                   help='One exact filename per line; same split for all models')
    p.add_argument('--groups', type=Path,
                   help='Optional CSV with filename,group; must cover every split item')
    p.add_argument('--output', type=Path, required=True,
                   help='New output directory, refuses overwriting')
    args = p.parse_args()
    names = [s.strip() for s in args.split.read_text(encoding='utf-8-sig').splitlines()
             if s.strip() and not s.lstrip().startswith('#')]
    if not names or len(names) != len(set(names)):
        raise ValueError('Split is empty or contains duplicates')
    for name in names:
        if '/' in name or '\\' in name or name in ('.', '..') or ':' in name:
            raise ValueError(f'Expected a filename, not a path: {name}')
    groups = {}
    if args.groups:
        with args.groups.open(encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                if row['filename'] in groups or not row['group'].strip():
                    raise ValueError('Duplicate filename or empty group')
                groups[row['filename']] = row['group'].strip()
        if set(names) - set(groups):
            raise ValueError('Groups CSV does not cover every evaluated image')
    rows = []
    for name in names:
        pred, truth = binary(args.predictions / name), binary(args.labels / name)
        if pred.shape != truth.shape:
            raise ValueError(f'Shape mismatch for {name}: {pred.shape} != {truth.shape}')
        tp = int(np.count_nonzero(pred & truth))
        fp = int(np.count_nonzero(pred & ~truth))
        fn = int(np.count_nonzero(~pred & truth))
        tn = int(np.count_nonzero(~pred & ~truth))
        rows.append(dict(filename=name, group=groups.get(name, 'all'),
                         tp=tp, fp=fp, fn=fn, tn=tn, **scores(tp, fp, fn, tn)))
    report = {
        'metric_unit': 'percent', 'sd_definition': 'sample SD across images, ddof=1',
        'undefined_policy': 'zero denominator -> null, excluded from macro; counts reported',
        'note': 'Descriptive image variation, not independent training-run uncertainty.',
        'split_sha256': hashlib.sha256(args.split.read_bytes()).hexdigest(),
        'predictions': str(args.predictions.resolve()), 'labels': str(args.labels.resolve()),
        'overall': summary(rows),
        'groups': {g: summary([r for r in rows if r['group'] == g])
                   for g in sorted({r['group'] for r in rows})}}
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'per_image.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (args.output / 'summary.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    with (args.output / 'table.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['group', 'n_images', 'metric', 'mean_percent', 'sd_percent', 'n_defined'])
        for group, values in [('overall', report['overall']), *report['groups'].items()]:
            for metric, s in values['macro_percent'].items():
                w.writerow([group, values['n_images'], metric, s['mean'], s['sd'], s['n_defined']])
    print(json.dumps(report['overall'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
