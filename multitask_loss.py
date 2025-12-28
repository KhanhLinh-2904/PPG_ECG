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