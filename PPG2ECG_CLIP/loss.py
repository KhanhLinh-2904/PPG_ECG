import torch
import torch.nn as nn
import torch.nn.functional as F

    
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_ecg, z_ppg):
        z_ecg = F.normalize(z_ecg, dim=1)
        z_ppg = F.normalize(z_ppg, dim=1)
        
        sim_matrix = torch.matmul(z_ecg, z_ppg.T) / self.temperature
        
        labels = torch.arange(z_ecg.size(0), dtype=torch.long, device=z_ecg.device)
        
        loss_ecg = F.cross_entropy(sim_matrix, labels)
        loss_ppg = F.cross_entropy(sim_matrix.T, labels)
        
        return (loss_ecg + loss_ppg) / 2

class PearsonLoss(nn.Module):
    def __init__(self):
        super(PearsonLoss, self).__init__()

    def forward(self, pred, target):
        pred = pred.squeeze(1)
        target = target.squeeze(1)
        
        pred_mean = pred.mean(dim=1, keepdim=True)
        target_mean = target.mean(dim=1, keepdim=True)
        
        pred_std = pred.std(dim=1, keepdim=True) + 1e-6
        target_std = target.std(dim=1, keepdim=True) + 1e-6
        
        cov = ((pred - pred_mean) * (target - target_mean)).mean(dim=1, keepdim=True)
        pearson_corr = cov / (pred_std * target_std)
        return 1 - pearson_corr.mean()