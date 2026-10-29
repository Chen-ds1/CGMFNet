import numpy as np


def extend_line(points, width, height):
    points = np.asarray(points, dtype=np.float64)
    if width < 2 or height < 2 or points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        return []
    if not np.isfinite(points).all() or np.ptp(points[:, 1]) < 1e-8:
        return []

    y = points[:, 1]
    yc = y-y.mean()
    a = float(np.dot(yc, points[:, 0]-points[:, 0].mean())/np.dot(yc, yc))
    b = float(points[:, 0].mean()-a*y.mean())
    xmax, ymax = width-1., height-1.
    candidates = [(b, 0.), (a*ymax+b, ymax)]
    if abs(a) > 1e-12:
        candidates.extend([(0., -b/a), (xmax, (xmax-b)/a)])
    valid = []
    for x, yy in candidates:
        if -1e-7 <= x <= xmax+1e-7 and -1e-7 <= yy <= ymax+1e-7:
            point = [float(np.clip(x, 0, xmax)), float(np.clip(yy, 0, ymax))]
            if not any(np.linalg.norm(np.asarray(point)-p) < 1e-7 for p in valid):
                valid.append(point)
    if len(valid) < 2:
        return []
    endpoints = max(((p,q) for i,p in enumerate(valid) for q in valid[i+1:]),
                    key=lambda pair: np.linalg.norm(np.asarray(pair[0])-pair[1]))
    return sorted(endpoints, key=lambda p: p[1], reverse=True)
