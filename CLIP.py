import torch
from torch import nn
import torch.nn.functional as F
import math
from ResNet50 import ResNet50_1D
from VisionTransformer import SignalTransformer

OUTPUT_EMBED_DIM = 128 
INPUT_LENGTH = 2400
BASE_WIDTH = 64
EXPANSION = 4 
LAYERS = [3, 4, 6, 3] 

class ECGEssembleCLIP(nn.Module):
    def __init__(self, embed_dim=128, ppg_input_channels=8): # <--- QUAN TRỌNG: Thêm tham số này
        super(ECGEssembleCLIP, self).__init__()
        
        self.encode_ecg = ResNet50_1D(input_channels=4, num_classes=embed_dim) #torch.Size([4, 4, 306])
        self.encode_ppg = ResNet50_1D(input_channels=ppg_input_channels, num_classes=embed_dim) #torch.Size([4, 8, 600])

        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def forward(self, ecg_original, ppg_original):
        # Forward pass giữ nguyên
        ecg_original_features, featured_ECG, feature_lists_ECG = self.encode_ecg(ecg_original)
        ecg_predicted_features, featured_PPG, feature_lists_PPG = self.encode_ppg(ppg_original)

        ecg_original_features = ecg_original_features / ecg_original_features.norm(dim=1, keepdim=True)
        ecg_predicted_features = ecg_predicted_features / ecg_predicted_features.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_original = logit_scale * ecg_original_features @ ecg_predicted_features.t()
        
        return logits_per_original, featured_PPG, feature_lists_PPG

class DecoderBlock_UNet(nn.Module):
    """Giữ nguyên Block cũ"""
    def __init__(self, in_channels, out_channels, skip_channels=0, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        total_in_channels = in_channels + skip_channels

        self.conv = nn.Sequential(
            nn.Conv1d(total_in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            # Resize x to match skip dimension exactly (xử lý lệch 1-2 pixel do padding)
            if x.size(2) != skip.size(2):
                x = nn.functional.interpolate(x, size=skip.size(2), mode='nearest')
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class ECGDecoder_DWT(nn.Module):
    """Decoder mới chuyên dùng để tái tạo DWT Features (kênh=4, dài=306)"""
    def __init__(self, bottleneck_channels=2048, output_channels=4, target_length=306):
        super().__init__()
        self.target_length = target_length

        self.adapter = nn.Sequential(
            nn.Conv1d(bottleneck_channels, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        ) 
        
        # Block 1: Input 512 + Skip f3 (1024) -> Output 256. (Upsample 38 -> ~75)
        self.block1 = DecoderBlock_UNet(in_channels=512, out_channels=256, skip_channels=1024)
        
        # Block 2: Input 256 + Skip f2 (512) -> Output 128. (Upsample 75 -> ~150)
        self.block2 = DecoderBlock_UNet(in_channels=256, out_channels=128, skip_channels=512)
        
        # Block 3: Input 128 + Skip f1 (256) -> Output 64. (Upsample 150 -> ~300)
        self.block3 = DecoderBlock_UNet(in_channels=128, out_channels=64, skip_channels=256)
        
        # Final Conv: Chuyển từ 64 kênh về 4 kênh (DWT channels)
        # Lưu ý: Không dùng Sigmoid ở cuối vì DWT coefficients không nằm trong đoạn [0,1]
        self.final_conv = nn.Conv1d(64, output_channels, kernel_size=1)

    def forward(self, z, features_list):
        # z: (B, 2048, ~38)
        # features_list: [f1(300), f2(150), f3(75)]
        
        f1, f2, f3 = features_list 
        
        x = self.adapter(z) 
        
        x = self.block1(x, skip=f3) # -> 75
        x = self.block2(x, skip=f2) # -> 150
        x = self.block3(x, skip=f1) # -> 300
        
        # Hiện tại x có chiều dài khoảng 300 (theo f1 của PPG ResNet).
        # Target là 306 (chiều dài DWT của ECG).
        # Ta nội suy lần cuối để khớp kích thước đích.
        if x.size(2) != self.target_length:
            x = nn.functional.interpolate(x, size=self.target_length, mode='linear', align_corners=False)
        
        x = self.final_conv(x)
        # Output: (Batch, 4, 306)
        return x


class PPGtoECGConverter(nn.Module):
    def __init__(self, embed_dim=OUTPUT_EMBED_DIM):
        super().__init__()
        self.ecg_decoder = ECGDecoder_DWT(
            bottleneck_channels=2048,
            output_channels=4,
            target_length=306
        )

    def forward(self, z_ppg, feature_lists_PPG):
        predicted_ecg_dwt = self.ecg_decoder(z_ppg, feature_lists_PPG)
        return predicted_ecg_dwt
    

# class ECGRegressionLayer(nn.Module):
#     def __init__(self, input_channels=4, input_len=306, target_len=2400):
#         super(ECGRegressionLayer, self).__init__()
        
#         # 1. Tính toán kích thước sau khi duỗi phẳng (Flatten)
#         # Input features shape: [Batch, 4, 306]
#         # Flatten dimension = 4 * 306 = 1224
#         self.flatten_dim = input_channels * input_len 
        
#         # 2. Dropout Layer (0.5) như trong hình vẽ
#         self.dropout = nn.Dropout(p=0.5)
        
#         # 3. Fully Connected Layer (Regression)
#         # Input: 1224 (features) -> Output: 2400 (ECG signal points)
#         self.fc = nn.Linear(self.flatten_dim, target_len)

#     def forward(self, x):
#         """
#         x: Predicted ECG features [Batch, 4, 306]
#         Returns: Reconstructed ECG [Batch, 2400]
#         """
#         # Lưu lại batch size (ví dụ: 49)
#         batch_size = x.size(0)
        
#         # Bước 1: Flatten
#         # [Batch, 4, 306] -> [Batch, 1224]
#         x = x.view(batch_size, -1)
        
#         # Bước 2: Dropout
#         x = self.dropout(x)
        
#         # Bước 3: Linear Regression để tái tạo tín hiệu
#         # [Batch, 1224] -> [Batch, 2400]
#         out = self.fc(x)
        
#         return out