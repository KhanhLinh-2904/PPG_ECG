import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

def pearson_corr_loss(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x.squeeze(1)
    y = y.squeeze(1)
    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)
    
    num = (x * y).sum(dim=1)
    den = torch.sqrt((x.pow(2).sum(dim=1) + eps) * (y.pow(2).sum(dim=1) + eps))
    
    corr = num / den
    return 1.0 - corr.mean()

class LearnableTemperature(nn.Module):
    def __init__(self, init_temp: float = 0.07):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / init_temp)))

    def forward(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100)

def clip_style_contrastive_loss(z_ppg: torch.Tensor, z_ecg: torch.Tensor, temperature_module: LearnableTemperature) -> torch.Tensor:
    z_ppg = F.normalize(z_ppg, dim=-1)
    z_ecg = F.normalize(z_ecg, dim=-1)
    scale = temperature_module()
    
    logits = scale * (z_ppg @ z_ecg.T)
    labels = torch.arange(z_ppg.size(0), device=z_ppg.device)
    
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    
    return 0.5 * (loss_i + loss_t)

class PPG2ECGLoss(nn.Module):

    def __init__(
        self, 
        init_temp: float = 0.07,
        lambda_recon: float = 1.0,
        lambda_pearson: float = 0.5,
        lambda_contrastive: float = 0.2
    ):
        super().__init__()
        self.temperature = LearnableTemperature(init_temp)
        self.lambda_recon = lambda_recon
        self.lambda_pearson = lambda_pearson
        self.lambda_contrastive = lambda_contrastive

    def forward(self, outputs: Dict[str, torch.Tensor], target_ecg: torch.Tensor) -> Dict[str, torch.Tensor]:
        recon = outputs["recon_ecg"]

        # 1. Loss reconstruction ECG (L1)
        loss_l1 = F.l1_loss(recon, target_ecg)
        
        # 2. Loss shape of reconstruction ECG (Pearson)
        loss_pearson = pearson_corr_loss(recon, target_ecg)

        # Total loss
        total_loss = (self.lambda_recon * loss_l1) + (self.lambda_pearson * loss_pearson)
        
        losses = {
            "loss_l1": loss_l1,
            "loss_pearson": loss_pearson,
        }

        # 3. Contrastive Loss 
        if "z_ppg" in outputs and "z_ecg" in outputs:
            loss_contrastive = clip_style_contrastive_loss(
                outputs["z_ppg"], 
                outputs["z_ecg"], 
                self.temperature
            )
            total_loss = total_loss + (self.lambda_contrastive * loss_contrastive)
            losses["loss_contrastive"] = loss_contrastive

        losses["loss_total"] = total_loss
        return losses