import numpy as np
from collections import deque
from scipy.spatial import cKDTree


def cluster_embeddings(embeddings, foreground, eps=1.0, min_samples=200):
    instances = np.zeros(foreground.shape, np.int32)
    vectors = embeddings[:, foreground].T
    if not len(vectors):
        return instances
    if eps <= 0 or min_samples < 1 or not np.isfinite(vectors).all():
        raise ValueError('Invalid DBSCAN parameters or non-finite embeddings')
    if len(vectors) < min_samples:
        return instances

    if np.linalg.norm(np.ptp(vectors.astype(np.float64), axis=0)) <= eps:
        instances[foreground] = 1
        return instances


    tree = cKDTree(vectors.astype(np.float64))
    core = tree.query_ball_point(vectors, eps, return_length=True) >= min_samples
    labels = np.zeros(len(vectors), np.int32)
    identity, remaining = 0, len(vectors)
    for seed in np.flatnonzero(core):
        if labels[seed]:
            continue
        identity += 1
        labels[seed] = identity
        remaining -= 1
        pending = deque([seed])
        while pending and remaining:
            point = pending.popleft()
            neighbors = np.asarray(tree.query_ball_point(vectors[point], eps), dtype=np.int64)
            unseen = neighbors[labels[neighbors] == 0]
            labels[unseen] = identity
            remaining -= len(unseen)
            pending.extend(unseen[core[unseen]])
        if remaining == 0:
            break
    instances[foreground] = labels
    return instances


def angle_error(k1, k2):
    angle = abs(np.degrees(np.arctan(k1) - np.arctan(k2))) % 180
    return float(min(angle, 180 - angle))


def fit_line(points, max_iterations=100, tolerance=0.01, epsilon=1e-3):
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2 or np.ptp(points[:, 1]) < 1e-6:
        return None
    x, y = points.T
    design = np.column_stack([y, np.ones_like(y)])
    coef = np.linalg.lstsq(design, x, rcond=None)[0]
    converged = False
    for step in range(max_iterations):
        residual = (x - design @ coef) / np.sqrt(1 + coef[0] ** 2)
        weight = 1.0 / np.maximum(0.5 * residual ** 2, epsilon)
        weight /= weight.max()
        root = np.sqrt(weight)
        updated = np.linalg.lstsq(design * root[:, None], x * root, rcond=None)[0]
        change = angle_error(coef[0], updated[0])
        coef = updated
        if change < tolerance:
            converged = True
            break
    return {'k': float(coef[0]), 'b': float(coef[1]), 'iterations': step + 1,
            'converged': converged, 'pixel_count': len(points)}


def extract_lines(instances):
    result = []
    for identity in np.unique(instances):
        if identity == 0:
            continue
        y, x = np.nonzero(instances == identity)
        line = fit_line(np.column_stack([x, y]))
        if line is not None:
            result.append(dict(instance_id=int(identity), **line))
    return sorted(result, key=lambda line: line['k'] * (instances.shape[0] - 1) + line['b'])
