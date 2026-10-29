import torch
from torch.nn import functional as F


def focal_loss(logits, target, alpha=0.75, gamma=2.0):
    target = target.to(logits.dtype)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    p = logits.sigmoid()
    pt = p * target + (1 - p) * (1 - target)
    weight = alpha * target + (1 - alpha) * (1 - target)
    return (weight * (1 - pt).pow(gamma) * ce).mean()


def discriminative_loss(embeddings, instances, delta_v=0.5, delta_d=1.5):
    variances, distances = [], []
    for emb, labels in zip(embeddings, instances):
        zero = emb.sum() * 0.0
        means, terms = [], []
        for identity in labels.unique():
            if identity.item() == 0:
                continue
            points = emb[:, labels == identity].T
            center = points.mean(0)
            means.append(center)
            terms.append((torch.linalg.vector_norm(points - center, dim=1) - delta_v).clamp_min(0).square().mean())
        variances.append(torch.stack(terms).mean() if terms else zero)
        if len(means) > 1:
            means = torch.stack(means)

            d = torch.linalg.vector_norm(means[:, None] - means[None, :], dim=-1)
            off_diagonal = ~torch.eye(len(means), device=d.device, dtype=torch.bool)
            distances.append((2 * delta_d - d[off_diagonal]).clamp_min(0).square().mean())
        else:
            distances.append(zero)
    return torch.stack(variances).mean(), torch.stack(distances).mean()


def total_loss(output, instances):
    focal = focal_loss(output['logits'], (instances > 0).unsqueeze(1))
    variance, distance = discriminative_loss(output['embeddings'], instances)
    return focal + variance + distance, {'focal': focal, 'variance': variance, 'distance': distance}
