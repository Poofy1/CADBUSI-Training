import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        probs = inputs
        pt = torch.where(targets == 1, probs, 1 - probs)
        
        # Apply alpha only to positive class
        alpha_factor = torch.where(targets == 1, self.alpha, 1.0)
        
        loss = -alpha_factor * (1 - pt).pow(self.gamma) * torch.log(pt + 1e-12)
        return loss.mean()
    
    
    
class BCELossWithSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        # Apply label smoothing
        target = target * (1 - self.smoothing) + 0.5 * self.smoothing
        # Calculate BCE
        bce = -(target * torch.log(pred + 1e-7) + (1 - target) * torch.log(1 - pred + 1e-7))
        return bce.mean()