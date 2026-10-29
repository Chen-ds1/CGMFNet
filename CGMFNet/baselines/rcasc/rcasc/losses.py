import torch
from torch.nn import functional as F

def criterion(out, loc, exist, c):
    p, e = out
    mask = loc != -100
    ce = F.cross_entropy(p[mask], loc[mask]) if mask.any() else p.sum() * 0
    expected = (p.softmax(-1) * torch.arange(c['bins'], device=p.device)).sum(-1)
    el = F.smooth_l1_loss(expected[mask], loc[mask].float()) if mask.any() else p.sum() * 0
    ex = F.cross_entropy(e.reshape(-1, 2), exist.reshape(-1), ignore_index=-100)
    return ce + c['lambda_expect'] * el + c['delta_exist'] * ex
