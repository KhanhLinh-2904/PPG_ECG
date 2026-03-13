import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import correlate
import numpy as np
import random
import matplotlib.pyplot as plt
import math
class PearsonCorrelationLoss(torch.nn.Module):
    def __init__(self):
        super(PearsonCorrelationLoss, self).__init__()

    def forward(self, x, y):
      
        x_flat = x.view(x.shape[0], -1)
        y_flat = y.view(y.shape[0], -1)
        
        mean_x = torch.mean(x_flat, dim=1, keepdim=True)
        mean_y = torch.mean(y_flat, dim=1, keepdim=True)
        
        xm = x_flat - mean_x
        ym = y_flat - mean_y
        
        r_num = torch.sum(xm * ym, dim=1)
        r_den = torch.sqrt(torch.sum(xm ** 2, dim=1) * torch.sum(ym ** 2, dim=1) + 1e-8)
        
        r = r_num / r_den
        return 1 - torch.mean(r)
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
        Shape: [Batch_Size, 1, Sequence_Length] hoặc [Batch_Size, Sequence_Length]
        """
        if ecg_batch.dim() == 3:
            ecg_batch = ecg_batch.squeeze(1)

        fft_vals = torch.fft.rfft(ecg_batch, dim=1)
        ecg_amp = torch.abs(fft_vals)
        ecg_amp = ecg_amp[:, 1:] 
        ecg_amp = torch.log1p(ecg_amp)

        ecg_norm = F.normalize(ecg_amp, p=2, dim=1)
        sim_matrix = torch.matmul(ecg_norm, ecg_norm.T)
        
        sim_matrix = sim_matrix ** 2 
        
        return sim_matrix

    def forward(self, student_logits, ecg_original):
        """
        student_logits: (Cosine similarity matrix)
        ecg_original:  [Batch, Length]
        """
        
        # 1. Student Log-Probabilities
        logits_student = student_logits / self.student_temp
        log_pred_row = F.log_softmax(logits_student, dim=1)
        log_pred_col = F.log_softmax(logits_student, dim=0)

        # 2. Teacher Probabilities 
        with torch.no_grad(): 
            target_sim_matrix = self.compute_target_similarity_gpu(ecg_original)
            # if random.random() < 0.01: # Thỉnh thoảng in ra 1 lần
            #     plt.imshow(target_sim_matrix.cpu().numpy(), cmap='viridis')
            #     plt.colorbar()
            #     plt.title("Teacher ECG Similarity Matrix (Raw Signal)")
            #     plt.show()
            
            logits_teacher = target_sim_matrix / self.teacher_temp
            target_prob_row = F.softmax(logits_teacher, dim=1)
            target_prob_col = F.softmax(logits_teacher, dim=0)

        # 3. KL Divergence Loss
        loss_row = F.kl_div(log_pred_row, target_prob_row, reduction='batchmean')
        loss_col = F.kl_div(log_pred_col, target_prob_col, reduction='batchmean')

        total_loss = (loss_row + loss_col) / 2
        return total_loss
    

def self_clustering_contrastive_loss(ecg_norm, ppg_norm, temperature=0.07):
    """
    Hàm Standard Contrastive Loss (InfoNCE) giúp xóa bỏ Modality Gap.
    """
    # 1. Tính ma trận Logits (Độ tương quan Cosine)
    # ecg_norm và ppg_norm đã được L2-normalized từ trước
    logits_per_ppg = (ppg_norm @ ecg_norm.t()) / temperature
    logits_per_ecg = (ecg_norm @ ppg_norm.t()) / temperature
    
    # 2. Tạo nhãn Ground Truth Tuyệt đối (Đường chéo chính)
    batch_size = ecg_norm.shape[0]
    # Tự động đẩy nhãn lên GPU/CPU khớp với device của input
    labels = torch.arange(batch_size, device=ecg_norm.device)
    
    # 3. Tính Cross Entropy Loss
    loss_ppg = F.cross_entropy(logits_per_ppg, labels)
    loss_ecg = F.cross_entropy(logits_per_ecg, labels)
    
    # 4. Trọng số trung bình
    total_loss = (loss_ppg + loss_ecg) / 2
    return total_loss