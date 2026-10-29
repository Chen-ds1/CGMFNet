import argparse
import json
from pathlib import Path
from tpisn.paths import DATA_DIR, PROJECT_DIR
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from tpisn.data import RiceRows
from tpisn.losses import total_loss
from tpisn.model import TPISN
from tpisn.calibration import calibrate_bn


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    total, count, tp, fp, fn = 0.0, 0, 0, 0, 0
    components = dict(focal=0.0, variance=0.0, distance=0.0)
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss, terms = total_loss(output, labels)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss; inspect images and instance IDs')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            n = len(images)
            total += loss.item() * n
            count += n
            for key, value in terms.items():
                components[key] += value.item() * n
            pred, gt = output['logits'][:, 0] > 0, labels > 0
            tp += (pred & gt).sum().item()
            fp += (pred & ~gt).sum().item()
            fn += (~pred & gt).sum().item()
    return dict(loss=total/count, iou=tp/max(tp+fp+fn, 1), f1=2*tp/max(2*tp+fp+fn, 1),
                **{key: value/count for key, value in components.items()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=str(DATA_DIR))
    p.add_argument('--output', default=str(PROJECT_DIR / 'runs' / 'tpisn'))
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--embedding-dim', type=int, default=4)
    p.add_argument('--channels', type=int, default=128)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--resume', help='Trusted checkpoint created by this project')
    p.add_argument('--no-bn-calibration', action='store_true', help='Disable training-only BN recalibration before validation')
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        p.error('epochs and batch-size must be positive')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    config = checkpoint['model_config'] if checkpoint else dict(embedding_dim=args.embedding_dim, channels=args.channels)
    model = TPISN(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start, best = 0, float('inf')
    generator = torch.Generator().manual_seed(args.seed)
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start, best = checkpoint['epoch'], checkpoint['best_val_loss']
        torch.set_rng_state(checkpoint['torch_rng'])
        generator.set_state(checkpoint['loader_rng'])
        if device.type == 'cuda' and checkpoint.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    loaders = {split: DataLoader(RiceRows(args.data, split), batch_size=args.batch_size,
               shuffle=split == 'train', num_workers=args.workers, generator=generator if split == 'train' else None)
               for split in ('train', 'val')}
    dest = Path(args.output)
    calibration_loader = DataLoader(RiceRows(args.data, 'train'), batch_size=args.batch_size,
                                   shuffle=False, num_workers=args.workers)
    dest.mkdir(parents=True, exist_ok=True)
    if not args.resume and (dest / 'last.pt').exists():
        p.error('Existing run: use --resume or a new --output directory')
    (dest / 'config.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
    for epoch in range(start, args.epochs):
        train = run_epoch(model, loaders['train'], device, optimizer)
        if not args.no_bn_calibration:
            calibrate_bn(model, calibration_loader, device)
        val = run_epoch(model, loaders['val'], device)
        improved = val['loss'] < best
        best = min(best, val['loss'])
        record = dict(epoch=epoch+1, train=train, val=val)
        with (dest / 'history.jsonl').open('a', encoding='utf-8') as file:
            file.write(json.dumps(record) + '\n')
        state = dict(epoch=epoch+1, model=model.state_dict(), model_config=model.config,
                     optimizer=optimizer.state_dict(), best_val_loss=best, train_args=vars(args),
                     torch_rng=torch.get_rng_state(), loader_rng=generator.get_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None)
        torch.save(state, dest / 'last.pt')
        if improved:
            torch.save(state, dest / 'best.pt')
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
