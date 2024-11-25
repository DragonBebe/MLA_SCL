import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    """Modified Cross-Entropy Loss inspired by supervised contrastive loss."""

    def __init__(self, temperature=0.07):
        """
        :param temperature: temperature scaling factor for logits
        """
        super(CrossEntropyLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Compute cross-entropy loss with temperature scaling.

        :param features: hidden vector of shape [batch_size, feature_dim]
        :param labels: ground truth labels of shape [batch_size]
        :return: computed cross-entropy loss
        """
        if len(features.shape) != 2:
            raise ValueError('`features` needs to be [batch_size, feature_dim]')

        batch_size = features.shape[0]
        device = features.device

        # Normalize features to unit sphere
        features = F.normalize(features, p=2, dim=1)

        # Compute similarity matrix
        logits = torch.matmul(features, features.T) / self.temperature

        # Apply mask to ignore self-comparisons
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        mask = torch.eye(batch_size, device=device, dtype=torch.bool)
        logits[mask] = float('-inf')

        # Compute probabilities
        exp_logits = torch.exp(logits)
        probabilities = exp_logits / (exp_logits.sum(dim=1, keepdim=True) + 1e-7)

        # Gather positive pairs for cross-entropy
        labels = labels.view(-1, 1)
        mask_pos = torch.eq(labels, labels.T).float().to(device)
        positive_logits = probabilities[mask_pos.bool()].view(batch_size, -1)

        # Compute loss
        log_probs = torch.log(positive_logits + 1e-7)
        loss = -log_probs.mean()

        return loss
