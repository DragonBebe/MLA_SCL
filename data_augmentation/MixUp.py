import torch
import numpy as np


def mixup_data(x, y, alpha=0.2):
    """
    Applies MixUp data augmentation to a batch of inputs and targets.

    Parameters:
        x (torch.Tensor): Batch of input images (shape: [batch_size, channels, height, width]).
        y (torch.Tensor): Batch of labels (shape: [batch_size, num_classes]).
        alpha (float): MixUp interpolation parameter.

    Returns:
        mixed_x (torch.Tensor): Batch of mixed images.
        y_a (torch.Tensor): Original labels for mixup loss calculation.
        y_b (torch.Tensor): Mixed labels for mixup loss calculation.
        lam (float): Lambda value for mixup loss calculation.
    """
    if alpha > 0:
        lam = max(np.random.beta(alpha, alpha), 0.1)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)  # Shuffle the indices

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Compute the mixup loss using a standard criterion.

    Parameters:
        criterion (function): Loss function, e.g., nn.CrossEntropyLoss.
        pred (torch.Tensor): Model predictions.
        y_a (torch.Tensor): Original labels.
        y_b (torch.Tensor): Mixed labels.
        lam (float): Lambda value for weighted loss calculation.

    Returns:
        loss (torch.Tensor): Computed loss.
    """
    
    assert pred.size(0) == y_a.size(0) == y_b.size(0), "Batch size mismatch between predictions and labels."
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
