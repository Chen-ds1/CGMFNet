import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    lovasz_hinge = None

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss']

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        probability = torch.sigmoid(input).reshape(input.size(0), -1)
        target = target.reshape(target.size(0), -1)
        intersection = (probability * target).sum(dim=1)
        dice = (2.0 * intersection + smooth) / (
            probability.sum(dim=1) + target.sum(dim=1) + smooth
        )
        dice_loss = 1.0 - dice.mean()
        return 0.5 * bce + 0.5 * dice_loss

class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        if lovasz_hinge is None:
            raise ImportError(
                'LovaszHingeLoss requires the optional LovaszSoftmax package.'
            )
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss
