import hashlib
import os
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from .utils import load, dump

def intersections(points, anchors, axis):
    p = np.asarray(points, float)
    out = np.full(len(anchors), -1.0, dtype=np.float32)
    for i, a in enumerate(anchors):
        hits = []
        overlap = False
        for u, v in zip(p[:-1], p[1:]):
            d = v[axis] - u[axis]
            if abs(d) < 1e-09:
                if abs(a - u[axis]) < 1e-07:
                    overlap = True
                continue
            t = (a - u[axis]) / d
            if -1e-07 <= t <= 1 + 1e-07:
                hits.append(float(u[1 - axis] + t * (v[1 - axis] - u[1 - axis])))
        hits = np.unique(np.round(hits, 6))
        if len(hits) == 1 and (not overlap) and (0 <= hits[0] <= 1):
            out[i] = hits[0]
        elif len(hits) > 1 or overlap:
            out[i] = -2.0
    return out

def targets(lines, c):
    a = (np.arange(c['anchors']) + 0.5) / c['anchors']
    vals = np.full((2, c['max_rows'], c['anchors']), -1.0, np.float32)
    for i, p in enumerate(lines):
        vals[0, i] = intersections(p, a, 1)
        vals[1, i] = intersections(p, a, 0)
    loc = np.clip(np.floor(np.maximum(vals, 0) * c['bins']), 0, c['bins'] - 1).astype(np.int64)
    exists = (vals >= 0).astype(np.int64)
    exists[vals == -2] = -100
    loc[vals < 0] = -100
    return (loc, exists)

def prepare(data, out, c):
    data = Path(data)
    records = []
    problems = []
    seen = {}
    for jp in sorted(data.glob('*.json')):
        d = load(jp)
        candidates = [jp.with_suffix('.png'), jp.with_suffix('.jpg'), jp.parent / d.get('imagePath', '')]
        ip = next((p for p in candidates if p.is_file()), None)
        if ip is None:
            raise ValueError(f'Missing image: {jp}')
        with Image.open(ip) as im:
            w, h = im.size
        if (w, h) != (d['imageWidth'], d['imageHeight']):
            raise ValueError(f'Size mismatch: {jp}')
        digest = hashlib.sha256(ip.read_bytes()).hexdigest()
        if digest in seen:
            raise ValueError(f'Duplicate image: {jp} and {seen[digest]}')
        seen[digest] = jp.name
        lines = []
        for s in d['shapes']:
            if s['shape_type'] not in ('line', 'linestrip'):
                raise ValueError(f'Unsupported shape: {jp}')
            p = np.asarray(s['points'], float)
            if len(p) < 2 or not np.isfinite(p).all():
                raise ValueError(f'Invalid points: {jp}')
            p = p / np.array([w - 1, h - 1])
            p = np.clip(p, 0, 1)
            lines.append(p.tolist())
        lines.sort(key=lambda p: float(np.asarray(p)[np.argmax(np.asarray(p)[:, 1]), 0]))
        if c['mode'] == 'core4':
            if len(lines) > 4:
                xs = [np.asarray(p)[np.argmax(np.asarray(p)[:, 1]), 0] for p in lines]
                k = min(range(len(lines) - 3), key=lambda k: abs((xs[k + 1] + xs[k + 2]) / 2 - 0.5))
                lines = lines[k:k + 4]
        if len(lines) > c['max_rows']:
            raise ValueError(f'Too many rows: {jp}')
        group = jp.stem[:11]
        val_groups = c.get('val_groups', ['20260601_11'])
        test_groups = c.get('test_groups', ['20260601_16'])
        if set(val_groups) & set(test_groups):
            raise ValueError('Validation/test groups overlap')
        split = 'test' if group in test_groups else 'val' if group in val_groups else 'train'
        records.append(dict(id=jp.stem, image=portable_path(ip, out), annotation=portable_path(jp, out), width=w, height=h, lines=lines, group=group, split=split, sha256=digest))
    if not records:
        raise ValueError('No LabelMe JSON files found')
    counts = {s: sum((r['split'] == s for r in records)) for s in ('train', 'val', 'test')}
    if min(counts.values()) == 0:
        raise ValueError('Default date split does not fit this dataset; edit split assignment')
    dump(records, Path(out) / 'manifest.json')
    dump(c, Path(out) / 'config.json')
    dump(dict(images=len(records), splits=counts, groups={s: sorted({r['group'] for r in records if r['split'] == s}) for s in counts}, row_counts={str(n): sum((len(r['lines']) == n for r in records)) for n in range(1, 10)}, issues=problems), Path(out) / 'audit.json')
    print(counts, flush=True)
    for r in records[::max(1, len(records) // 8)]:
        im = Image.open(Path(out) / r['image']).convert('RGB')
        draw = ImageDraw.Draw(im)
        for i, p in enumerate(r['lines']):
            xy = np.asarray(p) * [r['width'] - 1, r['height'] - 1]
            draw.line(list(map(tuple, xy)), fill='yellow', width=3)
            draw.text(tuple(xy[-1]), str(i + 1), fill='red')
        im.save(Path(out) / (r['id'] + '_labels.jpg'))

class Rows(Dataset):

    def __init__(self, records, c, augment=False):
        self.records = records
        self.c = c
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        im = Image.open(r['image']).convert('RGB')
        lines = r['lines']
        if self.augment:
            im = TF.adjust_brightness(im, random.uniform(0.85, 1.15))
            im = TF.adjust_contrast(im, random.uniform(0.85, 1.15))
            if random.random() < 0.5:
                im = TF.hflip(im)
                lines = [[[1 - x, y] for x, y in p] for p in reversed(lines)]
        x = TF.to_tensor(im.resize((self.c['width'], self.c['height']), Image.Resampling.BILINEAR))
        x = TF.normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        loc, exist = targets(lines, self.c)
        return (x, torch.from_numpy(loc), torch.from_numpy(exist))

def load_records(manifest, data_root=None):
    manifest = Path(manifest).resolve()
    records = load(manifest)
    for record in records:
        image = Path(record['image'])
        if data_root is not None:
            image = Path(data_root) / image.name
        elif not image.is_absolute():
            image = manifest.parent / image
        if not image.is_file() and data_root is None:
            parts = Path(str(record['image']).replace('\\', '/')).parts
            if 'my_labelme' in parts:
                suffix = Path(*parts[parts.index('my_labelme') + 1:])
                for parent in manifest.parents:
                    input_root = (parent / 'inputs').resolve()
                    candidate = (input_root / suffix).resolve()
                    if input_root in candidate.parents and candidate.is_file():
                        image = candidate
                        break
        if not image.is_file():
            raise FileNotFoundError(f'{image}: pass --data-root after moving the dataset')
        record['image'] = str(image.resolve())
    return records

def portable_path(path, base):
    path = Path(path).resolve()
    try:
        return os.path.relpath(path, Path(base).resolve())
    except ValueError:
        return str(path)
