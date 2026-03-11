import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class SoftTargetDistillationLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_embed, teacher_embed):
      
        student_embed = F.normalize(student_embed, dim=-1, p=2)
        teacher_embed = F.normalize(teacher_embed, dim=-1, p=2)

        sim_s_t = (student_embed @ teacher_embed.t()) / self.temperature
        
      
        with torch.no_grad(): 
            sim_t_t = (teacher_embed @ teacher_embed.t()) / self.temperature

        
        target_probs = F.softmax(sim_t_t, dim=-1)
        
        log_preds = F.log_softmax(sim_s_t, dim=-1)

        loss = F.kl_div(log_preds, target_probs, reduction='batchmean')

        return loss
    
class PearsonCorrelationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
     
        pred_mean = pred.mean(dim=1, keepdim=True)
        target_mean = target.mean(dim=1, keepdim=True)
        
        pred_centered = pred - pred_mean
        target_centered = target - target_mean
        
        cov = (pred_centered * target_centered).sum(dim=1)
        
        var_pred = pred_centered.pow(2).sum(dim=1).sqrt()
        var_target = target_centered.pow(2).sum(dim=1).sqrt()
        
        pearson_corr = cov / (var_pred * var_target + 1e-8)
        
        loss = 1.0 - pearson_corr.mean()
        return loss
    
class SpectralLoss(nn.Module):
    """ Hàm tính Loss trên miền Tần số (Thay thế hoàn hảo cho DTW) """
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        # QUAN TRỌNG: Ép kiểu dữ liệu về Float32 để tránh lỗi cuFFT không hỗ trợ Float16 
        # cho các mảng có chiều dài không phải là lũy thừa của 2 (như 2400)
        pred_fp32 = pred.to(torch.float32)
        target_fp32 = target.to(torch.float32)
        
        # Biến đổi sang miền tần số trên định dạng Float32 an toàn
        pred_fft = torch.fft.rfft(pred_fp32, dim=1)
        target_fft = torch.fft.rfft(target_fp32, dim=1)
        
        # Tính biên độ
        pred_amp = torch.abs(pred_fft)
        target_amp = torch.abs(target_fft)
        
        # Áp dụng L1 Loss
        loss_spectral = F.l1_loss(pred_amp, target_amp)
        return loss_spectral