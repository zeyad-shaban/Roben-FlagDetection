import torch.nn as nn
import torch


class WeightedDiceLoss(nn.Module):

    def __init__(self, foreground_weight=10.0, smooth=1.0):
        """
        :param foreground_weight: Weight for the foreground (object) class
        :param smooth: Smoothing constant to avoid division by zero
        """
        super().__init__()
        self.foreground_weight = foreground_weight
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()

        # Flatten predictions and labels
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)

        # Apply weights
        weights = torch.ones_like(y_true)
        weights[y_true == 1] = self.foreground_weight

        # Compute weighted intersection and union
        intersection = (weights * y_pred * y_true).sum()
        union = (weights * y_pred).sum() + (weights * y_true).sum()

        # Dice coefficient
        dsc = (2. * intersection + self.smooth) / (union + self.smooth)

        # Return 1 - DSC for loss
        return 1. - dsc
