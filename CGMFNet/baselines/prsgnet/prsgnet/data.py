import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def resolve_record_image(value, manifest):
    path = Path(value)
    candidate = path if path.is_absolute() else Path(manifest).resolve().parent / path
    if candidate.is_file():
        return candidate.resolve()
    parts = Path(str(value).replace('\\', '/')).parts
    if 'my_labelme' in parts:
        suffix = Path(*parts[parts.index('my_labelme') + 1:])
        for parent in Path(manifest).resolve().parents:
            input_root = (parent / 'inputs').resolve()
            updated = (input_root / suffix).resolve()
            if input_root in updated.parents and updated.is_file():
                return updated
    raise FileNotFoundError(f'Image not found in manifest {manifest}: {value}')


def encode_rows(rows, width, height, n=36):
    samples = np.linspace(0, height-1, n)
    xs, masks, geometries, lengths = [], [], [], []
    for row in rows:
        points = np.asarray(row, dtype=np.float32)
        if len(points) < 2 or not np.isfinite(points).all():
            continue
        points = points[np.argsort(points[:, 1])]
        yy, indices = np.unique(points[:, 1], return_index=True)
        if len(yy) < 2:
            continue
        xx = np.interp(samples, yy, points[indices, 0])[::-1].copy()
        valid = ((samples >= yy[0]) & (samples <= yy[-1]))[::-1].copy()
        valid &= (xx >= 0) & (xx <= width-1)
        if valid.sum() < 2:
            continue
        ids = np.flatnonzero(valid)

        first, last = ids[0], ids[-1]
        if not valid[first:last+1].all():
            continue
        dx = xx[last] - xx[first]
        dy = (last-first) * (height-1) / (n-1)
        theta = np.arctan2(dy, dx)
        geometries.append([xx[first]/(width-1), first/(n-1), theta])
        lengths.append(len(ids)/n)
        xs.append(xx/(width-1))
        masks.append(valid)
    return {'xs': torch.tensor(np.array(xs).reshape(-1, n), dtype=torch.float32),
            'valid': torch.tensor(np.array(masks).reshape(-1, n), dtype=torch.bool),
            'geometry': torch.tensor(geometries, dtype=torch.float32).reshape(-1, 3),
            'length': torch.tensor(lengths, dtype=torch.float32)}


def image_tensor(image):
    x = torch.from_numpy(np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1))).float()/255
    return (x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]


class RowDataset(Dataset):
    def __init__(self, manifest, augment=False, size=(320,800), n=36):
        self.path = Path(manifest)
        self.records = [json.loads(s) for s in self.path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
        for record in self.records:
            record['image'] = str(resolve_record_image(record['image'], self.path))
        self.augment, self.size, self.n = augment, size, n
        if not self.records:
            raise ValueError('Empty dataset manifest')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        rec = self.records[index]
        image = cv2.imread(str(self.path.parent / rec['image']))
        if image is None:
            raise FileNotFoundError(rec['image'])
        ih, iw = image.shape[:2]
        h, w = self.size
        rows = [np.asarray(row, np.float32).reshape(-1,2) * [(w-1)/(iw-1), (h-1)/(ih-1)] for row in rec['rows']]
        image = cv2.resize(image, (w,h))
        if self.augment:
            matrix = cv2.getRotationMatrix2D(((w-1)/2,(h-1)/2), np.random.uniform(-5,5), np.random.uniform(.9,1.1))
            matrix[:,2] += [np.random.uniform(-.05,.05)*w, np.random.uniform(-.05,.05)*h]
            image = cv2.warpAffine(image, matrix, (w,h))
            rows = [np.c_[r,np.ones(len(r))] @ matrix.T for r in rows]
            if np.random.rand() < .5:
                image = image[:,::-1].copy()
                rows = [np.c_[w-1-r[:,0],r[:,1]] for r in rows]
        target = encode_rows(rows,w,h,self.n)
        mask = np.zeros((h,w), np.uint8)
        for row in rows:
            if len(row)>1:
                cv2.polylines(mask,[np.rint(row).astype(np.int32)],False,1,30)
        target['seg'] = torch.from_numpy(mask).float().unsqueeze(0)
        return image_tensor(image), target


def collate(batch):
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)
