import torch
import torch.nn as nn

class WeightedHybridNetwork(nn.Module):
    def __init__(self, num_classes=2):
        super(WeightedHybridNetwork, self).__init__()

        # =========================================================
        # 1. MLP ĐƠN GIẢN CHỈ HỌC 3 FEATURES NHỊP TIM (tpr, rmssd, se)
        # =========================================================
        self.mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(16, 16),
            nn.ReLU(),
            
            nn.Linear(16, num_classes) # Xuất ra 2 giá trị thô (logits)
        )
        
        self.p_ratio_threshold = 0.7

    def forward(self, x):
        # 1. Tách dữ liệu đầu vào
        rr_features = x[:, :3]  # 3 tính năng nhịp tim
        p_ratio = x[:, 3:4]     # Tính năng P-ratio (cột cuối cùng)
        
        # =========================================================
        # 2. XÁC SUẤT TỪ MẠNG NEURAL (30% QUYỀN LỰC)
        # =========================================================
        mlp_logits = self.mlp(rr_features)
        # Ép về khoảng [0, 1] để thành Xác suất
        mlp_probs = torch.softmax(mlp_logits, dim=1) 
        
        # =========================================================
        # 3. XÁC SUẤT TỪ LUẬT THRESHOLD P_RATIO (70% QUYỀN LỰC)
        # =========================================================
        # Tạo tensor rỗng chứa xác suất của p_ratio
        pratio_probs = torch.zeros_like(mlp_probs)
        
        # Luật: Nếu p_ratio >= 0.7 -> 100% là Bình thường (Class 0)
        is_normal = (p_ratio >= self.p_ratio_threshold).float()
        # Ngược lại: p_ratio < 0.7 -> 100% là AFib (Class 1)
        is_af = 1.0 - is_normal
        
        pratio_probs[:, 0] = is_normal.squeeze() # Xác suất Class 0
        pratio_probs[:, 1] = is_af.squeeze()     # Xác suất Class 1
        
        # =========================================================
        # 4. HỢP NHẤT TỶ LỆ CHÍNH XÁC 70% - 30%
        # =========================================================
        final_probs = (0.00 * mlp_probs) + (1.0 * pratio_probs)
        
        final_logits = torch.log(final_probs + 1e-8)
        
        return final_logits