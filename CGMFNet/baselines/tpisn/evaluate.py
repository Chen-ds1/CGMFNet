import argparse
import json
from pathlib import Path
from tpisn.paths import DATA_DIR, PROJECT_DIR
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from tpisn.data import RiceRows, read_image
from tpisn.inference import load_model, predict
from tpisn.postprocess import angle_error


def reference_lines(annotation):
    groups = {}
    for line in annotation['lines']:
        groups.setdefault(line['instance_id'], []).extend(line['points'])
    result = []
    for points in groups.values():
        x, y = np.asarray(points, dtype=float).T
        if np.ptp(y) < 1e-6:
            continue
        k, b = np.linalg.lstsq(np.column_stack([y, np.ones_like(y)]), x, rcond=None)[0]
        result.append({'k': k, 'b': b})
    return result


def match_lines(pred, truth, gate=30.0):
    if not pred or not truth:
        return []
    ys = np.linspace(0, 255, 32)
    p = np.array([line['k'] * ys + line['b'] for line in pred])
    t = np.array([line['k'] * ys + line['b'] for line in truth])
    cost = np.abs(p[:, None] - t[None]).mean(-1)

    gated = np.where(cost <= gate, cost, (gate + 1) * (min(cost.shape) + 1))
    rows, cols = linear_sum_assignment(gated)
    return [angle_error(pred[i]['k'], truth[j]['k']) for i, j in zip(rows, cols) if cost[i, j] <= gate]


def main():
    p = argparse.ArgumentParser(description='Evaluate segmentation and matched row lines.')
    p.add_argument('--data', default=str(DATA_DIR))
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--split', default='test', choices=['val', 'test'])
    p.add_argument('--output', default=str(PROJECT_DIR / 'evaluation.json'))
    p.add_argument('--match-gate', type=float, default=30.0)
    p.add_argument('--cluster-method', choices=['fast', 'exact'], default='fast', help='fast: approximate anchor clustering; exact: DBSCAN')
    p.add_argument('--max-anchors', type=int, default=1024)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    dataset = RiceRows(args.data, args.split)
    model = load_model(args.checkpoint, args.device)
    tp = fp = fn = predicted = references = 0
    errors, per_image = [], []
    for name in dataset.ids:
        image = read_image(dataset.root / 'images' / (name + '.png'))
        labels = read_image(dataset.root / 'instances' / (name + '.png'), -1)
        prob, _, lines = predict(model, image, args.device, cluster_method=args.cluster_method, max_anchors=args.max_anchors)
        pred, gt = prob >= 0.5, labels > 0
        tp += int((pred & gt).sum())
        fp += int((pred & ~gt).sum())
        fn += int((~pred & gt).sum())
        annotation = json.loads((dataset.root / 'annotations' / (name + '.json')).read_text(encoding='utf-8'))
        truth = reference_lines(annotation)
        matched = match_lines(lines, truth, args.match_gate)
        predicted += len(lines)
        references += len(truth)
        errors.extend(matched)
        per_image.append(dict(id=name, predicted=len(lines), references=len(truth), matched=len(matched), errors_deg=matched))
    result = dict(split=args.split, cluster_method=args.cluster_method, max_anchors=args.max_anchors,
        images=len(dataset), iou=tp/max(tp+fp+fn, 1),
        f1=2*tp/max(2*tp+fp+fn, 1), matched_lines=len(errors), predicted_lines=predicted,
        reference_lines=references, line_precision=len(errors)/max(predicted, 1),
        line_recall=len(errors)/max(references, 1), me_deg=float(np.mean(errors)) if errors else None,
        rmse_deg=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
        match_gate_pixels=args.match_gate, per_image=per_image,
        note='ME/RMSE only on matched lines. Matching is an implementation choice; always report coverage.')
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'per_image'}, indent=2))


if __name__ == '__main__':
    main()
