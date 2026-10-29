import numpy as np


def fit_straight_rows(lines):
    result = []
    for line in lines:
        points = np.asarray([p for segment in line['segments'] for p in segment], dtype=float)
        if len(points) < 2 or not np.isfinite(points).all():
            continue
        independent = 1 if line['branch'] == 'row' else 0
        dependent = 1 - independent
        t, z = points[:, independent], points[:, dependent]
        low, high = 0.0, 1.0
        if float(np.ptp(t)) < 1e-8:
            continue
        a, b = np.linalg.lstsq(np.column_stack((t, np.ones_like(t))), z, rcond=None)[0]
        if abs(a) > 1e-12:
            bounds = sorted([-b / a, (1.0 - b) / a])
            low, high = max(low, bounds[0]), min(high, bounds[1])
        elif not 0 <= b <= 1:
            continue
        if high - low < 1e-8:
            continue
        endpoints = np.zeros((2, 2), dtype=float)
        endpoints[:, independent] = [low, high]
        endpoints[:, dependent] = a * endpoints[:, independent] + b
        fitted = dict(line)
        fitted['segments'] = [np.clip(endpoints, 0, 1).tolist()]
        result.append(fitted)
    return result
