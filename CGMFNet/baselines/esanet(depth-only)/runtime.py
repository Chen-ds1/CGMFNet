import csv
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader


def read_config(path):
    path = Path(path).resolve()
    c = json.loads(path.read_text(encoding='utf-8-sig'))
    for key in ('data_root', 'upstream', 'output_root', 'checkpoint'):
        if c.get(key):
            p = Path(c[key])
            c[key] = str(p if p.is_absolute() else path.parent / p)
    for split, spec in c.get('dataset_dirs', {}).items():
        if isinstance(spec, str):
            q = Path(spec)
            c['dataset_dirs'][split] = str(q if q.is_absolute() else path.parent / q)
        else:
            c['dataset_dirs'][split] = {
                k: str(Path(v) if Path(v).is_absolute() else path.parent / v)
                for k, v in spec.items() if v}
    return c


def names(path):
    items = [s.strip() for s in Path(path).read_text(encoding='utf-8-sig').splitlines()
             if s.strip() and not s.lstrip().startswith('#')]
    if not items or len(items) != len(set(items)):
        raise ValueError(f'Empty or duplicate split: {path}')
    if any('/' in n or '\\' in n or ':' in n or n in ('.', '..') for n in items):
        raise ValueError('Split must contain exact filenames, not paths')
    return items


def split_path(c, split):
    p = Path(c.get('splits', {}).get(split, f'splits/{split}.txt'))
    return p if p.is_absolute() else Path(c['data_root']) / p


def dataset_locations(c, split):
    spec = c.get('dataset_dirs', {}).get(split)
    if isinstance(spec, dict):
        required = ('depth', 'mask') if c['model'] == 'esanet_depth' else ('rgb', 'depth', 'mask')
        if any(not spec.get(k) for k in required):
            raise ValueError(f'Missing dataset directory for {split}: {required}')
        return {k: Path(v) for k, v in spec.items()}
    root = Path(spec or c['data_root'])
    return {'rgb': root / c['rgb_dir'], 'depth': root / c['depth_dir'], 'mask': root / c['mask_dir']}


def directory_mode(c, split):
    return bool(c.get('dataset_dirs', {}).get(split))


def selection_snapshot(c):
    result = {}
    for split in ('train', 'val', 'test'):
        locations = dataset_locations(c, split)
        if directory_mode(c, split):
            items = sorted(p.name for p in locations['mask'].glob('*.png'))
            result[split] = dict(directories={k: str(v) for k,v in locations.items()},
                                filenames=items, count=len(items),
                                filenames_sha256=hashlib.sha256(('\n'.join(items)).encode()).hexdigest())
        elif split_path(c, split).is_file():
            result[split] = dict(list_sha256=hashlib.sha256(split_path(c, split).read_bytes()).hexdigest())
    return result


def check_disjoint(a, b):

    for key in set(a.locations) & set(b.locations):
        def paths(ds):
            return {(ds.locations[key] / (Path(n).stem + ds.c['depth']['suffix'] if key == 'depth' else n)).resolve()
                    for n in ds.names}
        if paths(a) & paths(b):
            raise ValueError(f'Dataset partitions share the same {key} files')


def load_depth(path, spec):
    if spec.get('encoding') == 'realsense_pseudocolor':
        with Image.open(path) as im:
            a = np.array(im)
        if a.ndim != 3 or a.shape[2] != 3 or a.dtype != np.uint8:
            raise ValueError(f'Expected uint8 3-channel pseudocolor PNG: {path}')
        return a.astype(np.float32) / 255.0
    if not spec['confirmed_scalar_depth']:
        raise ValueError('Depth format unconfirmed. RealSense pseudocolor is not scalar depth. '
                         'Supply raw scalar depth, configure its units/range, then confirm in config.json.')
    if Path(path).suffix.lower() == '.npy':
        a = np.load(path, allow_pickle=False)
    else:
        with Image.open(path) as im:
            a = np.array(im)
    if a.ndim != 2:
        raise ValueError(f'Expected single-channel scalar depth, not pseudocolor: {path}, shape={a.shape}')
    a = a.astype(np.float32)
    valid = np.isfinite(a) & (a > spec['invalid_at_or_below'])
    lo, hi = spec['clip_min'], spec['clip_max']
    if hi <= lo or spec['scale'] <= 0 or spec['std'] <= 0:
        raise ValueError('Invalid depth scale, range, or standard deviation')
    a = np.clip(np.where(valid, a, 0) * spec['scale'], lo, hi)
    a = (a - lo) / (hi - lo)
    a = (a - spec['mean']) / spec['std']
    a[~valid] = spec['invalid_fill']
    return a


class RiceDataset(Dataset):
    def __init__(self, c, split, training=False, files=None):
        self.c, self.training = c, training
        self.locations = dataset_locations(c, split)
        if files is not None:
            self.names = files
        elif directory_mode(c, split):
            self.names = sorted(p.name for p in self.locations['mask'].glob('*.png'))
        else:
            self.names = names(split_path(c, split))
        if not self.names:
            raise ValueError(f'No PNG labels found for {split}: {self.locations["mask"]}')
        for name in self.names:
            required = [self.locations['mask'] / name,
                        self.locations['depth'] / (Path(name).stem + c['depth']['suffix'])]
            if c['model'] != 'esanet_depth':
                required.append(self.locations['rgb'] / name)
            for path in required:
                if not path.is_file():
                    raise FileNotFoundError(f'Missing paired image: {path}')

    def __len__(self):
        return len(self.names)

    def raw(self, index):
        name = self.names[index]
        dp = self.locations['depth'] / (Path(name).stem + self.c['depth']['suffix'])
        depth = load_depth(dp, self.c['depth'])
        rgb = None
        if self.c['model'] != 'esanet_depth':
            with Image.open(self.locations['rgb'] / name) as im:
                rgb = np.array(im.convert('RGB'))
            if rgb.shape[:2] != depth.shape[:2]:
                raise ValueError(f'Unaligned RGB/depth dimensions: {name}')
        return rgb, depth

    def inputs(self, rgb, depth):
        h, w = self.c['height'], self.c['width']
        d = torch.from_numpy(depth.copy())
        d = d.permute(2, 0, 1)[None] if depth.ndim == 3 else d[None, None]
        d = F.interpolate(d, size=(h, w), mode='nearest')[0]
        if rgb is None:
            r = torch.empty(0)
        else:
            r = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float()[None] / 255
            r = F.interpolate(r, size=(h, w), mode='bilinear', align_corners=False)[0]
            mean = torch.tensor(self.c['rgb_mean'])[:, None, None]
            std = torch.tensor(self.c['rgb_std'])[:, None, None]
            r = (r - mean) / std
        return r, d

    def __getitem__(self, index):
        rgb, depth = self.raw(index)
        with Image.open(self.locations['mask'] / self.names[index]) as im:
            mask = np.array(im)
        if mask.ndim != 2 or mask.shape != depth.shape[:2] or not set(np.unique(mask)).issubset({0, 1, 255}):
            raise ValueError(f'Invalid binary mask: {self.names[index]}')
        r, d = self.inputs(rgb, depth)
        y = torch.from_numpy((mask > 0).astype(np.int64))[None, None].float()
        y = F.interpolate(y, size=(self.c['height'], self.c['width']), mode='nearest')[0, 0].long()
        if self.training and random.random() < self.c['train']['horizontal_flip_probability']:
            if r.numel():
                r = r.flip(-1)
            d, y = d.flip(-1), y.flip(-1)
        return r, d, y


def build(c):
    from models import create
    if c['height'] % 32 or c['width'] % 32:
        raise ValueError('Input height and width must be multiples of 32')
    return create(c).to(c['device'])


def forward(model, c, rgb, depth):
    return model(depth) if c['model'] == 'esanet_depth' else model(rgb, depth)


def logits(out):
    return out[0] if isinstance(out, (tuple, list)) else out


def sync(c):
    if c['device'].startswith('cuda'):
        torch.cuda.synchronize(c['device'])


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def provenance(c):
    root = Path(__file__).parent
    return dict(config=c, implementation='local pseudocolor adaptation, random initialization',
                source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in root.rglob('*.py') if 'runs' not in p.relative_to(root).parts},
                torch=torch.__version__, cuda=torch.version.cuda, device=c['device'],
                gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                splits=selection_snapshot(c))


def new_output(c, name):
    p = Path(c['output_root']) / name
    p.mkdir(parents=True, exist_ok=False)
    return p


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def train(c):
    seed_all(c['seed'])
    ds, vs = RiceDataset(c, 'train', True), RiceDataset(c, 'val')
    check_disjoint(ds, vs)
    ds[0]; vs[0]
    t = c['train']
    if len(ds) < t['batch_size'] or t['batch_size'] < 2:
        raise ValueError('Use batch_size >= 2 and at least one full batch for training BatchNorm')
    dl = DataLoader(ds, batch_size=t['batch_size'], shuffle=True, num_workers=0, drop_last=True)
    vl = DataLoader(vs, batch_size=1, shuffle=False, num_workers=0)
    model = build(c)
    out = new_output(c, c['run_name'])
    meta = provenance(c)
    write_json(out / 'provenance.json', meta)
    opt = torch.optim.AdamW(model.parameters(), lr=t['lr'], weight_decay=t['weight_decay'])
    scaler = torch.cuda.amp.GradScaler(enabled=t['amp'] and c['device'].startswith('cuda'))
    best = -1.0
    with (out / 'history.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f); writer.writerow(['epoch', 'loss', 'val_foreground_iou_percent', 'lr'])
        for epoch in range(t['epochs']):
            lr = t['lr'] * (1 - epoch / t['epochs']) ** t['poly_power']
            for group in opt.param_groups:
                group['lr'] = lr
            model.train(); total = 0.
            for r, d, y in dl:
                r, d, y = r.to(c['device']), d.to(c['device']), y.to(c['device'])
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=c['device'].split(':')[0], enabled=scaler.is_enabled()):
                    outputs = forward(model, c, r, d)
                    outputs = outputs if isinstance(outputs, (list, tuple)) else [outputs]
                    loss = 0.
                    for i, v in enumerate(outputs):
                        v = F.interpolate(v, y.shape[-2:], mode='bilinear', align_corners=False)
                        loss = loss + (1 if i == 0 else t['aux_weight']) * F.cross_entropy(v, y)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                total += float(loss.detach())
            model.eval(); intersection = union = 0
            with torch.inference_mode():
                for r, d, y in vl:
                    v = logits(forward(model, c, r.to(c['device']), d.to(c['device'])))
                    p = F.interpolate(v, y.shape[-2:], mode='bilinear', align_corners=False).argmax(1).cpu().bool()
                    y = y.bool()
                    intersection += int((p & y).sum()); union += int((p | y).sum())
            if not union:
                raise ValueError('Undefined validation foreground IoU: no foreground union')
            iou = 100 * intersection / union
            writer.writerow([epoch + 1, total / len(dl), iou, lr]); f.flush()
            state = dict(state_dict=model.state_dict(), optimizer=opt.state_dict(), epoch=epoch + 1,
                         val_iou=iou, provenance=meta)
            torch.save(state, out / 'last.pth')
            if iou > best:
                best = iou
                torch.save(state, out / 'best.pth')
                torch.save(model.state_dict(), out / 'best_weights_only.pth')
            print(f'Epoch {epoch+1}/{t["epochs"]}: val IoU={iou:.3f}%, best={best:.3f}%')


def evaluate(c):
    seed_all(c['seed'])
    ds = RiceDataset(c, 'test')
    ds[0]

    ckpt = torch.load(c['checkpoint'], map_location='cpu', weights_only=False)
    trained = ckpt['provenance']['config']
    for key in ('model', 'architecture', 'height', 'width', 'rgb_mean', 'rgb_std', 'depth'):
        if c[key] != trained[key]:
            raise ValueError(f'Checkpoint/config mismatch: {key}')
    for split in ('train', 'val'):
        if directory_mode(c, split):
            loc = dataset_locations(c, split)
            if loc['mask'].is_dir():
                check_disjoint(ds, RiceDataset(c, split))
        elif split_path(c, split).exists():
            check_disjoint(ds, RiceDataset(c, split))
    model = build(c)
    model.load_state_dict(ckpt['state_dict'], strict=True); model.eval()
    out = new_output(c, c['evaluation']['name'])
    masks = out / 'masks'; masks.mkdir()
    amp = c['evaluation']['amp'] and c['device'].startswith('cuda')
    def infer(r, d):
        with torch.autocast(device_type=c['device'].split(':')[0], enabled=amp):
            return logits(forward(model, c, r, d))
    rows = []
    with torch.inference_mode():
        r, d, _ = ds[0]
        r, d = r[None].to(c['device']), d[None].to(c['device'])
        for _ in range(c['evaluation']['warmup']):
            infer(r, d)
        sync(c)
        for i, name in enumerate(ds.names):
            sync(c); start = time.perf_counter()
            rgb, depth = ds.raw(i)
            r, d = ds.inputs(rgb, depth)
            r, d = r[None].to(c['device']), d[None].to(c['device'])
            sync(c); before_model = time.perf_counter()
            v = infer(r, d)
            sync(c); after_model = time.perf_counter()
            mask = F.interpolate(v.float(), depth.shape[:2], mode='bilinear', align_corners=False).argmax(1)[0]
            mask = mask.cpu().numpy().astype(np.uint8) * 255
            sync(c); finished = time.perf_counter()
            Image.fromarray(mask).save(masks / name)
            model_ms = (after_model - before_model) * 1000
            pipe_ms = (finished - start) * 1000
            rows.append(dict(filename=name, model_ms=model_ms, segmentation_pipeline_ms=pipe_ms,
                             reciprocal_model_fps=1000/model_ms, reciprocal_pipeline_fps=1000/pipe_ms))
    with (out / 'timing_per_image.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    timing = {}
    for key in ('model_ms', 'segmentation_pipeline_ms'):
        values = np.array([r[key] for r in rows])
        timing[key] = dict(mean_ms=float(values.mean()), sd_ms=float(values.std(ddof=1)) if len(values)>1 else None,
                           aggregate_fps=float(1000 * len(values) / values.sum()),
                           reciprocal_fps_mean=float((1000/values).mean()),
                           reciprocal_fps_sd=float((1000/values).std(ddof=1)) if len(values)>1 else None)
    meta = provenance(c)
    meta.update(timing=timing, batch_size=1, precision='AMP' if amp else 'FP32',
                model_scope='forward only; CUDA synchronized; excludes transfer, resize, argmax, IO',
                pipeline_scope='read RGB/depth, preprocess, transfer, forward, original-size logits resize, argmax, CPU transfer; excludes mask saving, label reading, metrics; no row-line postprocessing',
                parameter_count=sum(p.numel() for p in model.parameters()),
                parameter_storage_MB=sum(p.numel()*p.element_size() for p in model.parameters())/1e6,
                checkpoint_file_MB=Path(c['checkpoint']).stat().st_size/1e6,
                checkpoint_sha256=hashlib.sha256(Path(c['checkpoint']).read_bytes()).hexdigest())
    write_json(out / 'runtime_summary.json', meta)
    evaluated_list = out / 'evaluated_files.txt'
    evaluated_list.write_text('\n'.join(ds.names) + '\n', encoding='utf-8')
    cmd = [sys.executable, str(Path(__file__).with_name('evaluate_masks.py')),
           '--predictions', str(masks), '--labels', str(ds.locations['mask']),
           '--split', str(evaluated_list), '--output', str(out/'metrics')]
    if c['evaluation']['groups_csv']:
        cmd.extend(['--groups', c['evaluation']['groups_csv']])
    subprocess.run(cmd, check=True)
    print(f'Saved masks, metrics, timing: {out}')


def smoke(c):
    seed_all(c['seed'])
    split = 'train' if directory_mode(c, 'train') else None
    locations = dataset_locations(c, split)
    files=sorted(p.name for p in locations['mask'].glob('*.png'))[:2]
    if len(files)<2:
        raise ValueError('Smoke requires at least two labeled samples')
    ds=RiceDataset(c,split,files=files)
    batch=[ds[i] for i in range(2)]
    r,d,y=[torch.stack([item[k] for item in batch]).to(c['device']) for k in range(3)]
    model=build(c)
    opt=torch.optim.AdamW(model.parameters(),lr=c['train']['lr'])
    scaler=torch.cuda.amp.GradScaler(enabled=c['train']['amp'] and c['device'].startswith('cuda'))
    losses=[]
    model.train()
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=c['device'].split(':')[0],enabled=scaler.is_enabled()):
            outputs=forward(model,c,r,d)
            outputs=outputs if isinstance(outputs,(tuple,list)) else [outputs]
            loss=sum((1 if i==0 else c['train']['aux_weight'])*F.cross_entropy(
                F.interpolate(v,y.shape[-2:],mode='bilinear',align_corners=False),y)
                for i,v in enumerate(outputs))
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite smoke loss')
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gradients=[p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise RuntimeError('Nonfinite or missing gradients')
        scaler.step(opt); scaler.update()
        losses.append(float(loss.detach()))
    model.eval()
    with torch.inference_mode():
        v=logits(forward(model,c,r[:1],d[:1]))
    if v.shape != (1,2,c['height'],c['width']) or not torch.isfinite(v).all():
        raise RuntimeError(f'Invalid prediction {v.shape}')
    out=new_output(c,'smoke_'+str(time.time_ns()))
    torch.save(model.state_dict(),out/'smoke_weights.pth')
    model.load_state_dict(torch.load(out/'smoke_weights.pth',map_location=c['device'],weights_only=True),strict=True)
    Image.fromarray(v.argmax(1)[0].cpu().numpy().astype(np.uint8)*255).save(out/'smoke_mask.png')
    report=dict(status='passed',model=c['model'],data_files=files,shape=list(v.shape),
                optimizer_steps=2,losses=losses,parameters=sum(p.numel() for p in model.parameters()),
                max_gpu_allocated_MB=torch.cuda.max_memory_allocated()/1e6 if torch.cuda.is_available() else None,
                note='Real-data execution check only, not convergence or held-out accuracy.',provenance=provenance(c))
    write_json(out/'smoke_report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='provenance'},indent=2))
    print('Report:',out/'smoke_report.json')
