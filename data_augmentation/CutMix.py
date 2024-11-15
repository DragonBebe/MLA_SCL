import torch
import numpy as np


def cutmix_data(x, y, alpha=1.0):
    """
    Applies CutMix data augmentation to a batch of inputs and targets.

    Parameters:
        x (torch.Tensor): Batch of input images (shape: [batch_size, channels, height, width]).
        y (torch.Tensor): Batch of labels (shape: [batch_size, num_classes]).
        alpha (float): CutMix interpolation parameter, controlling the size of the cut region.

    Returns:
        mixed_x (torch.Tensor): Batch of mixed images.
        y_a (torch.Tensor): Original labels for CutMix loss calculation.
        y_b (torch.Tensor): Mixed labels for CutMix loss calculation.
        lam (float): Lambda value for CutMix loss calculation.
    """
    batch_size, _, h, w = x.size()
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

    # Randomly choose a patch
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)

    # Uniformly sample the center of the patch
    cx = np.random.randint(w)
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    # Randomly choose another image in the batch
    index = torch.randperm(batch_size)
    x1, y1 = x[index], y[index]

    # Apply CutMix to the image
    x[:, :, bby1:bby2, bbx1:bbx2] = x1[:, :, bby1:bby2, bbx1:bbx2]

    # Adjust lambda to match the actual region
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (h * w))

    return x, y, y1, lam


def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    """
    Compute the CutMix loss using a standard criterion.

    Parameters:
        criterion (function): Loss function, e.g., nn.CrossEntropyLoss.
        pred (torch.Tensor): Model predictions.
        y_a (torch.Tensor): Original labels.
        y_b (torch.Tensor): Mixed labels.
        lam (float): Lambda value for weighted loss calculation.

    Returns:
        loss (torch.Tensor): Computed loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

