import torch
import torch.nn as nn

class FocusedNeuralNetwork(nn.Module):
    def __init__(self):
        super(FocusedNeuralNetwork, self).__init__()

        # Nhánh xử lý các đặc trưng RR (tpr, rmssd, entropy)
        self.rr_branch = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32), 
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Nhánh xử lý riêng cho p_ratio
        self.pratio_branch = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 32), 
            nn.ReLU()
        )

        self.regressor = nn.Sequential(
            nn.Linear(32 + 32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),            # Đầu ra nén về 1 chiều duy nhất đại diện cho ngưỡng
            nn.Sigmoid()                 # Ép giá trị ngưỡng về khoảng [0, 1] (Bỏ dòng này nếu ngưỡng > 1)
        )

    def forward(self, x):
        # x có shape: [batch_size, 4] với thứ tự [tpr, rmssd, entropy, p_ratio]
        
        # 1. Tách mảng dữ liệu
        rr_features = x[:, :3]   # Lấy 3 cột đầu tiên: [batch_size, 3]
        p_ratio = x[:, 3:4]      # Lấy cột cuối cùng: [batch_size, 1]

        # 2. Chạy qua 2 luồng riêng biệt
        out_rr = self.rr_branch(rr_features)         # Kích thước: [batch_size, 32]
        out_pratio = self.pratio_branch(p_ratio)     # Kích thước: [batch_size, 32]

        # 3. Hợp nhất (Concatenate) dọc theo chiều channels
        combined_features = torch.cat((out_rr, out_pratio), dim=1) # Kích thước: [batch_size, 64]

        # 4. Dự đoán và trả về giá trị ngưỡng (Threshold)
        threshold = self.regressor(combined_features) # Kích thước: [batch_size, 1]
        return threshold