import argparse
import base64
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from prsgnet.data import encode_rows

PROJECT = Path(__file__).resolve().parent


def read_image(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def resolve_image(annotation, doc, out):
    image_path = str(doc.get('imagePath') or '').replace('\\', '/')
    candidates = []
    if image_path:
        candidates += [annotation.parent / image_path,
                       annotation.parent / Path(image_path).name]
    candidates += [annotation.with_suffix(s) for s in ('.png', '.jpg', '.jpeg', '.bmp')]
    for path in candidates:
        if path.is_file():
            image = read_image(path)
            if image is None:
                raise ValueError(f'{annotation.name}: unreadable image {path}')
            return path.resolve(), image, 'file'
    if doc.get('imageData'):
        raw = base64.b64decode(doc['imageData'])
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'{annotation.name}: invalid embedded imageData')
        path = out / 'embedded_images' / (annotation.stem + '_' + hashlib.sha256(raw).hexdigest()[:12] + '.png')
        path.parent.mkdir(parents=True, exist_ok=True)
        ok, buffer = cv2.imencode('.png', image)
        if not ok:
            raise ValueError(f'Cannot encode image: {annotation}')
        buffer.tofile(str(path))
        return path.resolve(), image, 'embedded_imageData'
    raise FileNotFoundError(f'{annotation}: cannot resolve imagePath, sibling image, or imageData')


def make_group(stem, grouping):
    match = re.match(r'^(\d{8})_(\d{6})(?:_|$)', stem)
    if grouping == 'day':
        if not match:
            raise ValueError(f'--grouping day requires YYYYMMDD_HHMMSS filenames: {stem}')
        return match[1]
    return '_'.join(match.groups()) if match else stem


def convert(source, out, seed=42, labels=('rice_row',), grouping='capture', previews=True):
    source, out = Path(source).resolve(), Path(out).resolve()
    files = sorted(source.rglob('*.json'))
    if not files:
        raise FileNotFoundError(f'No LabelMe JSON files in {source}')
    records, audits, errors, warnings = [], [], [], []
    digest = hashlib.sha256()
    image_hashes = {}
    for annotation in files:
        try:
            raw = annotation.read_bytes()
            digest.update(str(annotation.relative_to(source)).encode('utf-8'))
            digest.update(raw)
            doc = json.loads(raw.decode('utf-8-sig'))
            image_path, image, origin = resolve_image(annotation, doc, out)
            height, width = image.shape[:2]
            if (doc.get('imageWidth'), doc.get('imageHeight')) != (width, height):
                raise ValueError('imageWidth/imageHeight disagree with actual image')
            rows, metadata = [], []
            shapes = doc.get('shapes', [])
            for index, shape in enumerate(shapes):
                if shape.get('label') not in labels:
                    continue
                if shape.get('shape_type') not in ('line', 'linestrip'):
                    raise ValueError(f'shape {index}: expected line/linestrip, got {shape.get("shape_type")}')
                points = np.asarray(shape['points'], dtype=np.float64)
                if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
                    raise ValueError(f'shape {index}: invalid points')
                dy = np.diff(points[:, 1])
                if not (np.all(dy > 0) or np.all(dy < 0)):
                    raise ValueError(f'shape {index}: row must be strictly monotone in y (no horizontal/folded segment)')
                if ((points < 0).any() or (points[:, 0] > width-1).any() or (points[:, 1] > height-1).any()):
                    raise ValueError(f'shape {index}: point outside image; correct annotation rather than silently clipping')
                rows.append(shape['points'])
                metadata.append({'shape_index': index, 'label': shape['label'], 'group_id': shape.get('group_id')})
            if shapes and not rows:
                raise ValueError(f'No matching line shapes for labels {labels}; refusing to turn this into a negative sample')
            target = encode_rows(rows, width, height)

            resized = [np.asarray(r) * [(800-1)/(width-1), (320-1)/(height-1)] for r in rows]
            if len(target['xs']) != len(rows) or len(encode_rows(resized, 800, 320)['xs']) != len(rows):
                raise ValueError('At least one row has fewer than two valid sampling heights; adjust annotation/representation')
            ids = [m['group_id'] for m in metadata if m['group_id'] is not None]
            duplicated = [i for i, count in Counter(ids).items() if count > 1]
            if duplicated:
                warnings.append({'file': str(annotation), 'duplicate_instance_group_ids': duplicated,
                                 'action': 'Each line/linestrip kept as an independent row; never merge by group_id.'})
            group = make_group(annotation.stem, grouping)
            ihash = hashlib.sha256(image.tobytes()).hexdigest()
            digest.update(ihash.encode())
            if ihash in image_hashes:
                raise ValueError(f'Duplicate image content with {image_hashes[ihash]}; resolve duplicate annotations before splitting')
            image_hashes[ihash] = str(annotation)
            record = {'image': str(image_path), 'rows': rows, 'group': group,
                      'label_source': 'manual_labelme', 'source_json': str(annotation), 'shapes_metadata': metadata}
            records.append(record)
            audits.append({'annotation': str(annotation), 'image': str(image_path), 'image_origin': origin,
                           'width': width, 'height': height, 'row_count': len(rows), 'encoded_rows': len(target['xs']),
                           'group': group, 'image_sha256': ihash,
                           'ignored_shapes': len(shapes)-len(rows)})
        except (ValueError, KeyError, TypeError, OSError) as exc:
            errors.append({'file': str(annotation), 'error': str(exc)})
    out.mkdir(parents=True, exist_ok=True)
    if errors:
        (out / 'conversion_errors.json').write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding='utf-8')
        raise ValueError(f'{len(errors)} invalid annotations. See {out / "conversion_errors.json"}. No manifests generated.')
    groups = sorted({r['group'] for r in records})
    if len(groups) < 3:
        raise ValueError('At least three capture/day groups are required for train/val/test.')
    np.random.default_rng(seed).shuffle(groups)
    nt = min(max(1, int(.7 * len(groups))), len(groups)-2)
    nv = min(max(1, int(.2 * len(groups))), len(groups)-nt-1)
    group_sets = {'train': set(groups[:nt]), 'val': set(groups[nt:nt+nv]), 'test': set(groups[nt+nv:])}
    splits = {name: [r for r in records if r['group'] in members] for name, members in group_sets.items()}
    for name, values in {**splits, 'all': records}.items():
        (out / (name + '.jsonl')).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in values), encoding='utf-8')
    if previews:
        preview = out / 'review'
        preview.mkdir(exist_ok=True)
        for index, rec in enumerate(records):
            image = read_image(rec['image'])
            for row_id, row in enumerate(rec['rows']):
                points = np.rint(row).astype(np.int32)
                color = ((row_id*73+80)%255, (row_id*127+180)%255, (row_id*43+240)%255)
                cv2.polylines(image, [points], False, color, 2)
                cv2.putText(image, str(row_id+1), tuple(points[len(points)//2]), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
            cv2.putText(image, 'MANUAL LABELME ROWS', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)
            ok, buffer = cv2.imencode('.png', image)
            if not ok:
                raise RuntimeError('Preview encoding failed')
            buffer.tofile(str(preview / f'{index:04d}_{Path(rec["source_json"]).stem}.png'))
    report = {'source': str(source), 'label_source': 'manual_labelme', 'labels': list(labels),
              'seed': seed, 'grouping': grouping, 'source_sha256': digest.hexdigest(),
              'images': len(records), 'rows': sum(len(r['rows']) for r in records), 'groups': len(groups),
              'split_counts': {k: len(v) for k,v in splits.items()},
              'split_rows': {k: sum(len(r['rows']) for r in v) for k,v in splits.items()},
              'split_groups': {k: sorted(v) for k,v in group_sets.items()}, 'warnings': warnings, 'files': audits,
              'split_note': 'Filename capture grouping prevents same-timestamp frames crossing splits; field/video independence is not established.'}
    (out/'audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('images', 'rows', 'groups', 'split_counts', 'split_rows')}, ensure_ascii=False), flush=True)
    return {name: out/(name+'.jsonl') for name in splits}


def main():
    from prsgnet.runtime import load_config, resolve
    config, config_path = load_config()
    p = argparse.ArgumentParser(description='Convert LabelMe row annotations into dataset manifests.')
    p.add_argument('--config', default=str(config_path))
    p.add_argument('--source', type=Path, default=resolve(config['data']['labelme_dir']))
    p.add_argument('--out', type=Path, default=resolve(config['data']['manifest_dir']))
    p.add_argument('--seed', type=int, default=config['data']['seed'])
    p.add_argument('--labels', nargs='+', default=config['data']['labels'])
    p.add_argument('--grouping', choices=['capture', 'day'], default=config['data']['grouping'])
    p.add_argument('--no-previews', action='store_true')
    a = p.parse_args()
    convert(resolve(a.source), resolve(a.out), a.seed, a.labels, a.grouping, not a.no_previews)


if __name__ == '__main__':
    main()
