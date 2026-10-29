import argparse
import base64
import json
from pathlib import Path
from tpisn.paths import INPUT_DIR, DATA_DIR
import random
import cv2
import numpy as np
from tpisn.data import labelme_mask, read_image, save_image


def main():
    p = argparse.ArgumentParser(description='Convert LabelMe line annotations into instance masks and dataset splits.')
    p.add_argument('--input', default=str(INPUT_DIR), help='Directory of Labelme JSON annotations')
    p.add_argument('--output', default=str(DATA_DIR))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--group-split', action='store_true', help='Split by parent folder (capture sequence), reducing video leakage')
    args = p.parse_args()
    root, dest = Path(args.input).resolve(), Path(args.output)
    annotations = sorted(root.rglob('*.json'))
    if len(annotations) < 3:
        p.error('At least three annotated images are needed for train/val/test')
    if dest.exists() and any(dest.iterdir()):
        p.error('Output must be empty; prevents mixing stale and new data splits')
    groups = {}
    for index, file in enumerate(annotations):
        ann = json.loads(file.read_text(encoding='utf-8-sig'))
        if ann.get('imageData'):
            image = cv2.imdecode(np.frombuffer(base64.b64decode(ann['imageData']), np.uint8), cv2.IMREAD_COLOR)
        else:
            image_path = Path(ann['imagePath'].replace('\\', '/'))
            resolved_image = file.parent / image_path
            if not resolved_image.is_file():
                resolved_image = file.parent / image_path.name
            image = read_image(resolved_image)
        if image is None or image.shape[:2] != (ann['imageHeight'], ann['imageWidth']):
            raise ValueError('Annotation/image dimensions disagree: ' + str(file))
        mask, lines = labelme_mask(ann)
        name = '{:06d}'.format(index)
        save_image(dest / 'images' / (name + '.png'), cv2.resize(image, (512, 256)))
        save_image(dest / 'instances' / (name + '.png'), mask)
        save_image(dest / 'semantic' / (name + '.png'), (mask > 0).astype(np.uint8) * 255)
        (dest / 'annotations').mkdir(exist_ok=True)
        (dest / 'annotations' / (name + '.json')).write_text(
            json.dumps({'source': file.relative_to(root).as_posix(), 'lines': lines}, ensure_ascii=False, indent=2), encoding='utf-8')
        group = str(file.parent.relative_to(root)) if args.group_split else name
        groups.setdefault(group, []).append(name)
    keys = sorted(groups)
    if len(keys) < 3:
        raise ValueError('Need at least three capture-sequence folders for --group-split')
    random.Random(args.seed).shuffle(keys)
    ntrain = min(len(keys) - 2, max(1, int(len(keys) * 0.6)))
    nval = min(len(keys) - ntrain - 1, max(1, int(len(keys) * 0.2)))
    (dest / 'splits').mkdir(exist_ok=True)
    counts = {}
    for split, selected in [('train', keys[:ntrain]), ('val', keys[ntrain:ntrain+nval]), ('test', keys[ntrain+nval:])]:
        ids = [name for key in selected for name in groups[key]]
        (dest / 'splits' / (split + '.txt')).write_text('\n'.join(ids) + '\n', encoding='utf-8')
        counts[split] = len(ids)
    (dest / 'metadata.json').write_text(json.dumps(dict(seed=args.seed, group_split=args.group_split,
        size=[512, 256], line_width=10, counts=counts), indent=2), encoding='utf-8')
    print(counts)


if __name__ == '__main__':
    main()
