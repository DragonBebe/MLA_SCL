# SupInt
# losses/supcon_loss_in.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss_in(nn.Module):
    """Supervised Contrastive Loss (SupConLoss_in).
    Implements the SupConLoss focusing on positive pairs within the sample set.
    """

    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        """
        :param temperature: temperature scaling factor for logits
        :param contrast_mode: 'all' (use all views as anchors) or 'one' (only use one view per sample as anchor)
        :param base_temperature: base temperature for scaling the loss
        """
        super(SupConLoss_in, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        Computes the SupCon_in loss, focusing on maximizing positive pair likelihood.

        :param features: hidden vector of shape [batch_size, n_views, feature_dim]
        :param labels: ground truth labels of shape [batch_size]
        :param mask: contrastive mask of shape [batch_size, batch_size], mask_{i,j}=1 if sample j
                     has the same class as sample i. If both labels and mask are None, it degenerates
                     to unsupervised contrastive loss.
        :return: computed SupCon_in loss
        """
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [batch_size, n_views, feature_dim], '
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)  # flatten feature dimensions

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            # If neither labels nor mask is provided, assume unsupervised contrastive loss
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            # Create mask based on labels
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        # Number of views and contrast features
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # [batch_size * n_views, feature_dim]

        # Choose anchor features based on contrast_mode
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]  # Only use the first view as anchor
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature  # Use all views as anchors
            anchor_count = contrast_count
        else:
            raise ValueError(f'Unknown contrast_mode: {self.contrast_mode}')

        # Compute logits and apply temperature scaling
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )

        # Numerical stability adjustment
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Repeat mask for multiple views if needed
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Compute log probabilities
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-7)

        # Compute mean log-likelihood for positive pairs, handling edge cases
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, torch.ones_like(mask_pos_pairs), mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # Loss calculation with scaling by temperature
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
