import torch
import torch.nn as nn


class CenterLoss(nn.Module):
    """
    Center loss from Wen et al. (ECCV 2016).
    Learns class centers in feature space and penalizes distances to target centers.
    """

    def __init__(self, num_classes: int, feat_dim: int):
        super().__init__()
        self.num_classes = int(num_classes)
        self.feat_dim = int(feat_dim)
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        # features: (B, D), labels: (B,)
        centers_batch = self.centers.index_select(0, labels)
        diff = features - centers_batch
        loss = 0.5 * (diff.pow(2).sum(dim=1)).mean()
        return loss
