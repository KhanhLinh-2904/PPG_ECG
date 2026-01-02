import torch
import torch.nn as nn

class MultiTaskLoss(nn.Module):
    def __init__(self, num_tasks=2):
        super(MultiTaskLoss, self).__init__()
        # Tạo tham số log_vars có thể học được (learnable parameters)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, c_loss, l_loss):
        # Task 1: Contrastive
        precision1 = torch.exp(-self.log_vars[0])
        loss1 = precision1 * c_loss + self.log_vars[0]

        # Task 2: MSE
        precision2 = torch.exp(-self.log_vars[1])
        loss2 = precision2 * l_loss + self.log_vars[1]

        return loss1 + loss2
    

class PearsonCorrelationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        # Tính trung bình trên trục thời gian (dim=2 hoặc dim=-1)
        x_mean = output.mean(dim=-1, keepdim=True)
        y_mean = target.mean(dim=-1, keepdim=True)
        
        # Trừ đi trung bình (Centering)
        x_centered = output - x_mean
        y_centered = target - y_mean
        
        # Tính tử số: Covariance
        numerator = (x_centered * y_centered).sum(dim=-1)
        
        # Tính mẫu số: Std * Std
        denominator = torch.sqrt((x_centered ** 2).sum(dim=-1)) * torch.sqrt((y_centered ** 2).sum(dim=-1))
        
        # Cộng thêm epsilon để tránh chia cho 0
        pearson = numerator / (denominator + 1e-8)
        
        # Loss là 1 - correlation (để càng giống nhau loss càng thấp)
        return 1 - pearson.mean()

# Tổng hợp Loss
class CombinedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.pearson = PearsonCorrelationLoss()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, output, target):
        loss_l1 = self.l1(output, target)
        loss_p = self.pearson(output, target)
        # L1 giúp đúng biên độ, Pearson giúp đúng hình dáng
        return self.alpha * loss_l1 + self.beta * loss_p