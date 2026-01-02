import torch
import torch.nn as nn
class NegPearsonLoss(nn.Module):
    def __init__(self):
        super(NegPearsonLoss, self).__init__()

    def forward(self, preds, targets):
        # Flatten về (Batch, Length)
        preds = preds.view(preds.shape[0], -1)
        targets = targets.view(targets.shape[0], -1)
        
        # Tính mean
        mean_preds = torch.mean(preds, dim=1, keepdim=True)
        mean_targets = torch.mean(targets, dim=1, keepdim=True)
        
        # Trừ mean
        preds_centered = preds - mean_preds
        targets_centered = targets - mean_targets
        
        # Tính Correlation
        num = torch.sum(preds_centered * targets_centered, dim=1)
        denom = torch.sqrt(torch.sum(preds_centered ** 2, dim=1)) * torch.sqrt(torch.sum(targets_centered ** 2, dim=1))
        
        pearson = num / (denom + 1e-8)
        
        # Loss = 1 - correlation (để tối ưu hóa tiến về 1)
        return 1 - torch.mean(pearson)