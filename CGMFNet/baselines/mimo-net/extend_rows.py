from pathlib import Path
import config as C
import json
import numpy as np
from PIL import Image, ImageDraw

EVALUATION_DIR = None
RGB_DIR = C.DATA_ROOT / 'rgb'
OUTPUT_DIR = EVALUATION_DIR

def boundary_line(points, width, height):
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2 or not np.isfinite(p).all():
        return None
    if np.ptp(p[:, 1]) > 1e-8:

        a, b = np.polyfit(p[:, 1], p[:, 0], 1)
        anchor = np.array([b, 0.0]); direction = np.array([a, 1.0])
    else:
        if np.ptp(p[:, 0]) < 1e-8: return None
        anchor = p.mean(0); direction = np.array([1.0, 0.0])
    low, high = -np.inf, np.inf
    for k, bound in enumerate([width - 1, height - 1]):
        if abs(direction[k]) < 1e-12:
            if not 0 <= anchor[k] <= bound: return None
        else:
            t = sorted([(0-anchor[k])/direction[k], (bound-anchor[k])/direction[k]])
            low, high = max(low, t[0]), min(high, t[1])
    if low >= high or not np.isfinite([low, high]).all(): return None
    return np.clip(np.array([anchor + low*direction, anchor + high*direction]), [0,0], [width-1,height-1])

def render_full(original, lines, raw_tracks=None):
    overlay = original.convert('RGB').copy(); draw = ImageDraw.Draw(overlay)
    full = []
    for i, line in enumerate(lines):
        points = raw_tracks[i] if raw_tracks is not None and i < len(raw_tracks) else line
        extended = boundary_line(points, overlay.width, overlay.height)
        if extended is None: continue
        full.append({'source_index': i, 'points': extended.tolist()})
        draw.line([tuple(v) for v in extended], fill=(255,0,0), width=2)
    return overlay, full

def rebuild(evaluation_dir=EVALUATION_DIR, rgb_dir=RGB_DIR, output_dir=OUTPUT_DIR):
    if evaluation_dir is None:
        candidates = [p for p in C.RUN_ROOT.glob('evaluation_*') if (p/'lines').is_dir()]
        if not candidates:
            raise FileNotFoundError('No evaluation results found; run evaluate.py or set EVALUATION_DIR')
        evaluation_dir = max(candidates, key=lambda p: (p.stat().st_mtime_ns, str(p)))
    output_dir = evaluation_dir if output_dir is None else output_dir
    evaluation_dir, rgb_dir, output_dir = map(Path, [evaluation_dir, rgb_dir, output_dir])
    files = sorted((evaluation_dir/'lines').glob('*.json'))
    if not files: raise FileNotFoundError(f'No predicted line JSON: {evaluation_dir / "lines"}')
    for folder in ['overlays_full', 'lines_full']:
        (output_dir/folder).mkdir(parents=True, exist_ok=True)
    counts = {'images': 0, 'input_lines': 0, 'extended_lines': 0}
    for path in files:
        data = json.loads(path.read_text(encoding='utf-8'))
        with Image.open(rgb_dir/(path.stem+'.png')) as im:
            if im.size != (data['width'], data['height']): raise ValueError(f'Size mismatch: {path.stem}')
            overlay, full = render_full(im, data['lines'], data.get('raw_tracks'))
        overlay.save(output_dir/'overlays_full'/(path.stem+'.png'))
        (output_dir/'lines_full'/path.name).write_text(json.dumps({
            'purpose': 'boundary extrapolation for visualization only; original metrics unchanged',
            'width': data['width'], 'height': data['height'], 'lines': full}, indent=2), encoding='utf-8')
        counts['images'] += 1; counts['input_lines'] += len(data['lines']); counts['extended_lines'] += len(full)
    (output_dir/'extension_report.json').write_text(json.dumps(counts,indent=2),encoding='utf-8')
    print(counts); print(output_dir/'overlays_full')
    return counts

if __name__ == '__main__': rebuild()
