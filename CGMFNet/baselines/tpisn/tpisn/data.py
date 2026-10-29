import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def read_image(path, flags=cv2.IMREAD_COLOR):
    result = cv2.imdecode(np.fromfile(str(path), np.uint8), flags)
    if result is None:
        raise ValueError('Cannot read image: ' + str(path))
    return result


def save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError('Cannot encode: ' + str(path))
    data.tofile(str(path))


def image_tensor(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    return torch.from_numpy(rgb.transpose(2, 0, 1).copy())


class RiceRows(Dataset):
    def __init__(self, root, split):
        self.root = Path(root)
        self.ids = (self.root / 'splits' / (split + '.txt')).read_text(encoding='utf-8').split()
        if not self.ids:
            raise ValueError('Empty split: ' + split)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        name = self.ids[index]
        image = read_image(self.root / 'images' / (name + '.png'))
        labels = read_image(self.root / 'instances' / (name + '.png'), cv2.IMREAD_UNCHANGED)
        if image.shape[:2] != (256, 512) or labels.shape != (256, 512):
            raise ValueError('Expected 512x256 images and single-channel instance labels: ' + name)
        return image_tensor(image), torch.from_numpy(labels.astype(np.int64)), name


def labelme_mask(annotation, width=512, height=256, line_width=10):
    mask = np.zeros((height, width), np.uint16)
    groups = {}
    sx, sy = width / annotation['imageWidth'], height / annotation['imageHeight']
    lines = []
    for index, shape in enumerate(annotation['shapes']):
        if shape.get('shape_type') not in ('linestrip', 'line'):
            raise ValueError('Only line/linestrip annotations are accepted; found ' + str(shape.get('shape_type')))
        points = np.asarray(shape['points'], dtype=np.float64)
        if len(points) < 2 or not np.isfinite(points).all():
            raise ValueError('A row must contain at least two finite points')
        points = points * [sx, sy]
        group = shape.get('group_id')
        key = ('group', group) if group is not None else ('shape', index)
        identity = groups.setdefault(key, len(groups) + 1)

        cv2.polylines(mask, [np.rint(points).astype(np.int32)], False, int(identity), line_width, cv2.LINE_8)
        lines.append({'instance_id': identity, 'points': points.tolist()})
    return mask, lines
