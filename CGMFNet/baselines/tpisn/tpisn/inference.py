import cv2
import numpy as np
import torch
from .data import image_tensor
from .model import TPISN
from .postprocess import cluster_embeddings, extract_lines
from .fast_postprocess import cluster_embeddings_fast, extract_lines_fast


def load_model(checkpoint, device):


    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = TPISN(**saved['model_config'])
    model.load_state_dict(saved['model'])
    return model.to(device).eval()


@torch.no_grad()
def predict(model, image, device, threshold=0.5, eps=1.0, min_samples=200,
            cluster_method='fast', max_anchors=1024):
    resized = cv2.resize(image, (512, 256))
    output = model(image_tensor(resized).unsqueeze(0).to(device))
    probability = output['logits'][0, 0].sigmoid().cpu().numpy()
    embeddings = output['embeddings'][0].cpu().numpy()
    if cluster_method == 'fast':
        instances = cluster_embeddings_fast(embeddings, probability >= threshold, eps, min_samples, max_anchors)
    elif cluster_method == 'exact':
        instances = cluster_embeddings(embeddings, probability >= threshold, eps, min_samples)
    else:
        raise ValueError('cluster_method must be fast or exact')
    lines = extract_lines_fast(instances) if cluster_method == 'fast' else extract_lines(instances)
    return probability, instances, lines


def original_lines(lines, width, height):
    sx, sy = width / 512.0, height / 256.0
    return [dict(line, k=line['k'] * sx / sy, b=line['b'] * sx) for line in lines]


def overlay(image, instances, lines):
    out = image.copy()
    labels = cv2.resize(instances.astype(np.float32), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    for line in lines:
        identity = line['instance_id']
        color = np.random.default_rng(identity).integers(60, 256, size=3).astype(np.uint8)
        mask = labels == identity
        out[mask] = (0.5 * out[mask] + 0.5 * color).astype(np.uint8)
        h, w = out.shape[:2]
        a = (int(round(line['b'])), 0)
        b = (int(round(line['k'] * (h-1) + line['b'])), h-1)
        visible, a, b = cv2.clipLine((0, 0, w, h), a, b)
        if visible:
            cv2.line(out, a, b, tuple(map(int, color)), 2, cv2.LINE_AA)
    return out
