import torch
import torch.nn as nn
import torch.nn.functional as F

class DistributionAlignmentLoss(nn.Module):
    """
    Lớp Loss dùng để căn chỉnh phân phối đặc trưng (Feature Distribution Alignment)
    giữa Source (PPG) và Target (ECG).
    """
    def __init__(self, eps: float = 1e-8):
        super(DistributionAlignmentLoss, self).__init__()
        # Epsilon nhỏ giúp ổn định số học, tránh lỗi chia cho 0 hoặc log(0)
        self.eps = eps

    def coral_loss(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """ 
        Tính CORAL trên toàn bộ chiều dữ liệu của batch [B, C*L].
        Giảm thiểu sự khác biệt về ma trận hiệp phương sai (Covariance) giữa 2 miền.
        """
        source = source.flatten(1)
        target = target.flatten(1)

        source = source - source.mean(dim=0, keepdim=True)
        target = target - target.mean(dim=0, keepdim=True)

        bs = source.size(0)
        bt = target.size(0)

        cov_source = (source.T @ source) / max(bs - 1, 1)
        cov_target = (target.T @ target) / max(bt - 1, 1)

        d = source.size(1)
        return torch.sum((cov_source - cov_target) ** 2) / (4.0 * d * d + self.eps)

    def kl_divergence_batch(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """ 
        Tính KL Divergence dựa trên thống kê Batch Mean và Batch Variance.
        Giả định các đặc trưng tuân theo phân phối Gaussian đơn biến.
        """
        source = source.flatten(1)
        target = target.flatten(1)
        
        # Thống kê Mean & Var của PPG (Source)
        mean_p = source.mean(dim=0)
        var_p = source.var(dim=0, unbiased=False) + self.eps
        
        # Thống kê Mean & Var của ECG (Target) - Đóng băng Gradient
        mean_e = target.mean(dim=0).detach()
        var_e = target.var(dim=0, unbiased=False).detach() + self.eps
        
        # Công thức KL Divergence cho 2 phân phối Gaussian đơn biến (tính cho từng chiều)
        kl = 0.5 * (torch.log(var_e / var_p) + (var_p + (mean_p - mean_e)**2) / var_e - 1.0)
        
        return torch.mean(kl)

    def forward(self, source: torch.Tensor, target: torch.Tensor, alpha_coral: float = 1.0, beta_kl: float = 1.0) -> dict:
        """
        Tính toán và kết hợp các thành phần loss.
        
        Args:
            source (torch.Tensor): Đặc trưng đầu ra từ PPG [B, C, L] hoặc [B, D]
            target (torch.Tensor): Đặc trưng đầu ra từ ECG [B, C, L] hoặc [B, D]
            alpha_coral (float): Trọng số cho CORAL Loss
            beta_kl (float): Trọng số cho KL Divergence Loss
            
        Returns:
            dict: Dictionary chứa tổng loss và các thành phần chi tiết
        """
        coral_val = self.coral_loss(source, target)
        kl_val = self.kl_divergence_batch(source, target)
        
        total_loss = (alpha_coral * coral_val) + (beta_kl * kl_val)
        
        return {
            "total_loss": total_loss,
            "coral_loss": coral_val,
            "kl_loss": kl_val
        }
    
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