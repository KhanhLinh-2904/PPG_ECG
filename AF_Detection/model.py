import torch
import torch.nn as nn

class FocusedNeuralNetwork(nn.Module):
    def __init__(self, num_classes=2):
        super(FocusedNeuralNetwork, self).__init__()

        # =================================================================
        # LUỒNG 1: Xử lý sự hỗn loạn của nhịp tim (3 features: tpr, rmssd, entropy)
        # =================================================================
        self.rr_branch = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32), # Nén xuống còn 32 chiều
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # =================================================================
        # LUỒNG 2: Khuếch đại đặc trưng sóng P (1 feature: p_ratio)
        # =================================================================
        # Thay vì đi chung, p_ratio đi đường VIP. 
        # Biến 1 con số thành 32 con số (Tạo ra không gian biểu diễn rộng hơn)
        self.pratio_branch = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 32), # Khuếch đại lên 32 chiều (Ngang hàng với luồng RR)
            nn.ReLU()
        )

        # =================================================================
        # LỚP QUYẾT ĐỊNH (FINAL CLASSIFIER)
        # =================================================================
        # Nhận vào: 32 nơ-ron từ nhịp tim + 32 nơ-ron từ P-ratio
        self.classifier = nn.Sequential(
            nn.Linear(32 + 32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        # x có shape: [batch_size, 4] với thứ tự [tpr, rmssd, entropy, p_ratio]
        
        # 1. Tách mảng dữ liệu
        rr_features = x[:, :3]   # Lấy 3 cột đầu tiên
        p_ratio = x[:, 3:4]      # Lấy cột cuối cùng (giữ nguyên định dạng ma trận 2D)

        # 2. Chạy qua 2 luồng riêng biệt
        out_rr = self.rr_branch(rr_features)         # Kích thước: [batch_size, 32]
        out_pratio = self.pratio_branch(p_ratio)     # Kích thước: [batch_size, 32]

        # 3. Hợp nhất (Concatenate)
        # Lúc này "thế lực" của P-ratio đã ngang bằng với cả 3 tính năng kia gộp lại
        combined_features = torch.cat((out_rr, out_pratio), dim=1) # Kích thước: [batch_size, 64]

        # 4. Đưa ra phán quyết cuối cùng
        return self.classifier(combined_features)