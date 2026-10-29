import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from .postprocess import angle_error


def cluster_embeddings_fast(embeddings, foreground, eps=1.0, min_samples=200, max_anchors=1024):
    instances = np.zeros(foreground.shape, np.int32)
    vectors = np.ascontiguousarray(embeddings[:, foreground].T, dtype=np.float64)
    n = len(vectors)
    if eps <= 0 or min_samples < 1 or max_anchors < 1 or not np.isfinite(vectors).all():
        raise ValueError('Invalid clustering parameters or embeddings')
    if n < min_samples:
        return instances
    if np.linalg.norm(np.ptp(vectors, axis=0)) <= eps:
        instances[foreground] = 1
        return instances

    indices = np.linspace(0, n - 1, min(n, max_anchors), dtype=np.int64)
    anchors = vectors[indices]
    tree = cKDTree(anchors)
    distances, nearest = tree.query(vectors, k=1, workers=1)
    weights = np.bincount(nearest, minlength=len(anchors))
    adjacent = cdist(anchors, anchors, metric='sqeuclidean') <= eps ** 2
    density = adjacent @ weights
    core = density >= min_samples
    if not core.any():
        return instances
    graph = csr_matrix(adjacent[core][:, core])
    _, labels = connected_components(graph, directed=False)


    anchor_labels = np.zeros(len(anchors), dtype=np.int32)
    anchor_labels[core] = labels + 1
    assigned = anchor_labels[nearest]


    needs_core = ~core[nearest]
    if needs_core.any():
        d, idx = cKDTree(anchors[core]).query(vectors[needs_core], k=1, workers=1)
        distances[needs_core] = d
        assigned[needs_core] = labels[idx] + 1
    assigned[distances > eps] = 0
    instances[foreground] = assigned
    return instances


def fit_line_fast(points, max_iterations=100, tolerance=0.01, epsilon=1e-3):
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2 or np.ptp(points[:, 1]) < 1e-6:
        return None
    x, y = points.T
    xm, ym = x.mean(), y.mean()
    yc = y - ym
    k = np.sum(yc * (x - xm)) / np.sum(yc * yc)
    b = xm - k * ym
    converged = False
    for step in range(max_iterations):
        residual = (x - (k*y + b)) / np.sqrt(1 + k*k)
        weights = 1.0 / np.maximum(0.5 * residual*residual, epsilon)
        weights /= weights.max()
        total = weights.sum()
        ym = np.sum(weights*y) / total
        xm = np.sum(weights*x) / total
        yc = y - ym
        denominator = np.sum(weights*yc*yc)
        if denominator <= np.finfo(float).tiny:
            break
        updated = np.sum(weights*yc*(x-xm)) / denominator
        change = angle_error(k, updated)
        k, b = updated, xm - updated*ym
        if change < tolerance:
            converged = True
            break
    return dict(k=float(k), b=float(b), iterations=step+1, converged=converged, pixel_count=len(points))


def extract_lines_fast(instances):
    lines = []
    for identity in np.unique(instances):
        if identity == 0:
            continue
        y, x = np.nonzero(instances == identity)
        line = fit_line_fast(np.column_stack([x, y]))
        if line is not None:
            lines.append(dict(instance_id=int(identity), **line))
    return sorted(lines, key=lambda line: line['k']*(instances.shape[0]-1)+line['b'])
