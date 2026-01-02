import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import correlate
import numpy as np

class SoftCLIPLoss(nn.Module):
    def __init__(self, teacher_temp=0.05, student_temp=0.07):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
    
    def get_max_cross_correlation_score(self, x, y):
        if isinstance(x, torch.Tensor):
           
            x = x.detach().cpu().numpy().flatten()
            
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy().flatten()
        corr = correlate(x, y, mode='full')
        
        return np.max(corr)
    def compute_similarity_matrix(self, ecg_array):
       
        n_segments = ecg_array.shape[0]
        sim_matrix = np.zeros((n_segments, n_segments))
        
       
        for i in range(n_segments):
            for j in range(i, n_segments):
                if i == j:
                    sim_matrix[i, j] = 1.0 
                else:
                    score = self.get_max_cross_correlation_score(ecg_array[i], ecg_array[j])
                    sim_matrix[i, j] = score
                    sim_matrix[j, i] = score 
                    
        return sim_matrix 
    
    def forward(self, similarity_matrix, ecg_array):

        logits_student = similarity_matrix / self.student_temp
        log_pred_row = F.log_softmax(logits_student, dim=1)
        log_pred_col = F.log_softmax(logits_student, dim=0)


        ecg_cross_corr_matrix = self.compute_similarity_matrix(ecg_array)
        target_matrix = torch.tensor(ecg_cross_corr_matrix, dtype=torch.float32).to(logits_student.device)
        logits_teacher = target_matrix / self.teacher_temp
        target_prob_row = F.softmax(logits_teacher, dim=1)
        target_prob_col = F.softmax(logits_teacher, dim=0)
        
        # Loss = KL(Target || Prediction)
        # nn.KLDivLoss nhận input là log-probabilities và target là probabilities
        loss_row = F.kl_div(log_pred_row, target_prob_row, reduction='batchmean')
        loss_col = F.kl_div(log_pred_col, target_prob_col, reduction='batchmean')
        
        total_loss = (loss_row + loss_col) / 2
        return total_loss


class FastSoftCLIPLoss(nn.Module):
    def __init__(self, teacher_temp=0.05, student_temp=0.07):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp

    def compute_target_similarity_gpu(self, ecg_batch):
        """
        Tính Similarity Matrix cho ECG batch hoàn toàn trên GPU.
        Giả sử ecg_batch đã được chuẩn hóa (normalized) từ bước Preprocessing.
        Shape: [Batch_Size, 1, Sequence_Length] hoặc [Batch_Size, Sequence_Length]
        """
        # Đảm bảo input là 2D: [Batch, Length]
        if ecg_batch.dim() == 3:
            ecg_batch = ecg_batch.squeeze(1)
            
        # 1. Chuẩn hóa vector (L2 Norm) để tích vô hướng trở thành Cosine Similarity
        # Cross-Correlation ở lag 0 của 2 tín hiệu đã chuẩn hóa chính là Cosine Similarity
        ecg_norm = F.normalize(ecg_batch, p=2, dim=1)
        
        # 2. Tính ma trận tương đồng bằng phép nhân ma trận (Matrix Multiplication)
        # [Batch, Length] x [Length, Batch] -> [Batch, Batch]
        sim_matrix = torch.matmul(ecg_norm, ecg_norm.T)
        
        # Lưu ý: Nếu bạn muốn Max Cross-Correlation (cho phép lệch pha - lag), 
        # thuật toán sẽ phức tạp hơn (dùng FFT). 
        # Tuy nhiên, do bước Preprocessing bạn đã align PPG theo ECG, 
        # nên Cosine Similarity (Lag 0) là đủ tốt và cực nhanh.
        
        return sim_matrix

    def forward(self, student_logits, ecg_original):
        """
        student_logits: output từ model (Cosine similarity matrix của features)
        ecg_original: Tín hiệu ECG gốc [Batch, Length]
        """
        
        # 1. Tính Student Log-Probabilities
        logits_student = student_logits / self.student_temp
        log_pred_row = F.log_softmax(logits_student, dim=1)
        log_pred_col = F.log_softmax(logits_student, dim=0)

        # 2. Tính Teacher Probabilities (Dựa trên độ giống nhau của tín hiệu gốc)
        with torch.no_grad(): # Không tính gradient cho Teacher
            target_sim_matrix = self.compute_target_similarity_gpu(ecg_original)
            
            # Xử lý các giá trị 1.0 trên đường chéo (chính nó so với nó)
            # SoftCLIP thường giữ nguyên, để model học rằng mẫu i giống mẫu i nhất.
            
            logits_teacher = target_sim_matrix / self.teacher_temp
            target_prob_row = F.softmax(logits_teacher, dim=1)
            target_prob_col = F.softmax(logits_teacher, dim=0)

        # 3. Tính KL Divergence Loss
        loss_row = F.kl_div(log_pred_row, target_prob_row, reduction='batchmean')
        loss_col = F.kl_div(log_pred_col, target_prob_col, reduction='batchmean')

        total_loss = (loss_row + loss_col) / 2
        return total_loss