import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import correlate
import numpy as np

class SoftCLIPLoss(nn.Module):
    def __init__(self, teacher_temp=0.1, student_temp=0.07):
        """
        Args:
            teacher_temp: Nhiệt độ cho ma trận cross-correlation (Ground Truth).
                          Càng nhỏ thì distribution càng nhọn (gần one-hot).
            student_temp: Nhiệt độ cho output của model CLIP.
        """
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
    
    def get_max_cross_correlation_score(self, x, y):
        if isinstance(x, torch.Tensor):
           
            x = x.detach().cpu().numpy().flatten()
            
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy().flatten()
        # -------------------------------------------------
        # Tính cross-correlation
        corr = correlate(x, y, mode='full')
        
        # Lấy giá trị lớn nhất
        return np.max(corr)
    def compute_similarity_matrix(self, ecg_array):
        """
        Tạo ma trận N x N từ mảng ECG (N, Length).
        """
        n_segments = ecg_array.shape[0]
        sim_matrix = np.zeros((n_segments, n_segments))
        
        # Duyệt qua từng cặp (i, j)
        # Vì ma trận đối xứng (A giống B thì B giống A), ta chỉ cần tính 1 nửa
        for i in range(n_segments):
            for j in range(i, n_segments):
                if i == j:
                    sim_matrix[i, j] = 1.0 # Tự so sánh với chính nó
                else:
                    score = self.get_max_cross_correlation_score(ecg_array[i], ecg_array[j])
                    sim_matrix[i, j] = score
                    sim_matrix[j, i] = score # Đối xứng
                    
        return sim_matrix 
    
    def forward(self, similarity_matrix, ecg_array):
        """
        Args:
            ecg_cross_corr_matrix: (Batch, Batch) Ma trận chứa giá trị max_cross_corr 
                                   giữa các ECG trong batch. Giá trị từ -1 đến 1.
        """
        
        ecg_cross_corr_matrix = self.compute_similarity_matrix(ecg_array)
        # logits: (Batch, Batch)
        logits = similarity_matrix / self.student_temp
        
        # Log-Softmax theo chiều hàng (cho PPG -> ECG) và cột (cho ECG -> PPG)
        log_pred_row = F.log_softmax(logits, dim=1)
        log_pred_col = F.log_softmax(logits, dim=0)

        # 2. Xây dựng Target Distribution (Teacher) từ Cross-Correlation
        # Đảm bảo target matrix nằm trên cùng device với logits
        target_matrix = torch.tensor(ecg_cross_corr_matrix, dtype=torch.float32).to(logits.device)
        
        # Áp dụng Softmax lên Target Matrix để tạo xác suất (Probability Distribution)
        # Lưu ý: Cross-corr có thể âm, nhưng exp() sẽ xử lý thành dương.
        target_prob_row = F.softmax(target_matrix / self.teacher_temp, dim=1)
        target_prob_col = F.softmax(target_matrix / self.teacher_temp, dim=0)

        # 3. Tính KL Divergence Loss
        # Loss = KL(Target || Prediction)
        # nn.KLDivLoss nhận input là log-probabilities và target là probabilities
        loss_row = F.kl_div(log_pred_row, target_prob_row, reduction='batchmean')
        loss_col = F.kl_div(log_pred_col, target_prob_col, reduction='batchmean')
        
        # Tổng hợp loss hai chiều (Symmetric Loss)
        total_loss = (loss_row + loss_col) / 2
        return total_loss

