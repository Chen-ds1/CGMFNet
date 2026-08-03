import numpy as np
import torch

def iou_score(output, target, threshold=0.5, smooth=1e-5):

    if torch.is_tensor(output):
        output = torch.sigmoid(output).detach().cpu().numpy()
    else:
        output = np.asarray(output)

    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    else:
        target = np.asarray(target)

    output_binary = output > threshold
    target_binary = target > 0.5

    if output_binary.ndim < 2:
        raise ValueError(
            f"Expected batched masks, got shape {output_binary.shape}."
        )

    reduce_axes = tuple(range(1, output_binary.ndim))
    intersection = np.logical_and(
        output_binary, target_binary
    ).sum(axis=reduce_axes)
    union = np.logical_or(
        output_binary, target_binary
    ).sum(axis=reduce_axes)

    return float(np.mean((intersection + smooth) / (union + smooth)))

def dice_coef(output, target, smooth=1e-5):
    output = torch.sigmoid(output).reshape(output.size(0), -1)
    target = target.reshape(target.size(0), -1)
    intersection = (output * target).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (
        output.sum(dim=1) + target.sum(dim=1) + smooth
    )
    return float(dice.mean().detach().cpu())
