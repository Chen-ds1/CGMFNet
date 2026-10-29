import random
import math
import time
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import Rows, load_records
from .model import RCASC
from .losses import criterion
from .evaluation import evaluate
from .utils import load, dump

def train(args):
    root = Path(args.prepared)
    c = load(args.config or root / 'config.json')
    records = load_records(root / 'manifest.json', args.data_root)
    if args.epochs:
        c['epochs'] = args.epochs
    if args.batch_size:
        c['batch_size'] = args.batch_size
    random.seed(c['seed'])
    np.random.seed(c['seed'])
    torch.manual_seed(c['seed'])
    torch.set_num_threads(4)
    device = args.device if args.device != 'auto' else 'cuda' if torch.cuda.is_available() else 'cpu'
    model = RCASC(c, pretrained=c['pretrained'] and (not args.resume)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=c['lr'], betas=(0.9, 0.999), weight_decay=c['weight_decay'])
    tr = [r for r in records if r['split'] == 'train']
    va = [r for r in records if r['split'] == 'val']
    loader = DataLoader(Rows(tr, c, True), batch_size=c['batch_size'], shuffle=True, num_workers=0)
    out = Path(args.output)
    if (out / 'last.pt').exists() and (not args.resume):
        raise FileExistsError('Output contains a run; choose a new output or use --resume')
    if not tr or not va:
        raise ValueError('Training and validation subsets must be non-empty')
    out.mkdir(parents=True, exist_ok=True)
    best = -1
    history = []
    start = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        for key in c:
            if key not in ('epochs', 'batch_size') and ck['config'].get(key) != c[key]:
                raise ValueError(f'Incompatible resume config: {key}')
        if not (out / 'best.pt').exists():
            raise ValueError('Resume into the original run directory, preserving best.pt')
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['optimizer'])
        start = ck['epoch']
        best = ck['best_f1']
        history = ck.get('history', [])
        if start >= c['epochs']:
            raise ValueError('Total epochs must exceed checkpoint epoch')
    dump(c, out / 'config.json')
    dump(dict(torch=torch.__version__, device=device, gpu=torch.cuda.get_device_name(0) if device == 'cuda' else None), out / 'environment.json')
    for epoch in range(start, c['epochs']):
        begin = time.perf_counter()
        model.train()
        factor = min(1.0, (epoch + 1) / c['warmup']) * (0.5 * (1 + math.cos(math.pi * max(0, epoch - c['warmup']) / max(1, c['epochs'] - c['warmup']))))
        for g in opt.param_groups:
            g['lr'] = c['lr'] * factor
        total = 0
        for x, l, e in loader:
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x.to(device)), l.to(device), e.to(device), c)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            opt.step()
            total += loss.item() * len(x)
        metric = evaluate(model, va, c, device)
        improved = metric['f1'] > best
        if improved:
            best = metric['f1']
        item = dict(epoch=epoch + 1, train_loss=total / len(tr), val_f1=metric['f1'], lr=opt.param_groups[0]['lr'], seconds=time.perf_counter() - begin)
        history.append(item)
        ck = dict(model=model.state_dict(), optimizer=opt.state_dict(), config=c, epoch=epoch + 1, best_f1=best, history=history)
        torch.save(ck, out / 'last.pt')
        if improved:
            torch.save(ck, out / 'best.pt')
        dump(history, out / 'history.json')
        print(json.dumps(item), flush=True)
    model.load_state_dict(torch.load(out / 'best.pt', map_location=device, weights_only=False)['model'])
    print('val', evaluate(model, va, c, device, out / 'val'), flush=True)
