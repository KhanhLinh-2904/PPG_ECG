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
    def __init__(self, embed_dim=OUTPUT_EMBED_DIM):
        super(ECGEssembleCLIP, self).__init__()

        self.encode_ecg_resnet = ResNet50_1D(
        layers=[3, 4, 6, 3] ,
        num_classes=OUTPUT_EMBED_DIM)

        self.encode_ecg_transformer =  SignalTransformer(input_length=INPUT_LENGTH,
        patch_size=30,
        width=512,
        layers=6,
        heads=8,
        output_dim=OUTPUT_EMBED_DIM)

        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def forward(self, ecg_original, ppg_original):

        ecg_original_features = self.encode_ecg_resnet(ecg_original)
        ecg_predicted_features = self.encode_ecg_transformer(ppg_original)

        ecg_original_features = ecg_original_features / ecg_original_features.norm(dim=1, keepdim=True)
        ecg_predicted_features = ecg_predicted_features / ecg_predicted_features.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        
       
        logits_per_original = logit_scale * ecg_original_features @ ecg_predicted_features.t()
        
        logits_per_predicted = logits_per_original.t()

        return logits_per_original, logits_per_predicted


class DecoderBlock(nn.Module):
    """Một khối Upsampling + Conv1D để giải mã."""
    def __init__(self, in_channels, out_channels, kernel_size=3, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        # Dùng padding='same' để giữ nguyên độ dài sau conv
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

class ECGDecoder(nn.Module):
    """
    Bộ giải mã ECG: Nhận vector (B, 128) -> Tín hiệu (B, 1, 2400)
    """
    def __init__(self, embed_dim=128, start_channels=512, start_length=75):
        super().__init__()
        self.start_length = start_length
        self.start_channels = start_channels
        
        # 1. Biến vector (B, 128) thành 1 chuỗi ngắn (B, 512, 75)
        self.initial_project = nn.Linear(embed_dim, start_channels * start_length)
        self.initial_bn = nn.BatchNorm1d(start_channels)
        self.initial_relu = nn.ReLU()

        # 2. Các khối Upsampling
        # 75 -> 150 (channels: 512 -> 256)
        self.block1 = DecoderBlock(start_channels, 256)
        # 150 -> 300 (channels: 256 -> 128)
        self.block2 = DecoderBlock(256, 128)
        # 300 -> 600 (channels: 128 -> 64)
        self.block3 = DecoderBlock(128, 64)
        # 600 -> 1200 (channels: 64 -> 32)
        self.block4 = DecoderBlock(64, 32)
        # 1200 -> 2400 (channels: 32 -> 16)
        self.block5 = DecoderBlock(32, 16)

        # 3. Lớp Conv cuối cùng để ra 1 channel (tín hiệu ECG)
        self.final_conv = nn.Conv1d(16, 1, kernel_size=1)
        # Dùng Tanh để ép giá trị đầu ra về [-1, 1] (nếu bạn đã chuẩn hoá)
        self.final_activation = nn.Tanh()

    def forward(self, z):
        # z có shape (B, 128)
        x = self.initial_project(z) # (B, 512 * 75)
        
        # Reshape thành (B, 512, 75)
        x = x.view(x.size(0), self.start_channels, self.start_length)
        x = self.initial_bn(x)
        x = self.initial_relu(x)
        
        x = self.block1(x) # (B, 256, 150)
        x = self.block2(x) # (B, 128, 300)
        x = self.block3(x) # (B, 64, 600)
        x = self.block4(x) # (B, 32, 1200)
        x = self.block5(x) # (B, 16, 2400)
        
        x = self.final_conv(x) # (B, 1, 2400)
 
        return x

# class DecoderBlock(nn.Module):
#     """
#     A simple decoder block that uses Upsampling + Conv.
#     This block reverses the spatial downsampling of a corresponding encoder block.
#     """
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(DecoderBlock, self).__init__()
#         self.stride = stride
        
#         # Upsampling layer
#         self.upsample = nn.Upsample(scale_factor=stride, mode='nearest') if stride > 1 else nn.Identity()
        
#         # 1x1 conv for the shortcut to match channels and apply after upsampling
#         self.shortcut_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
#         self.shortcut_bn = nn.BatchNorm1d(out_channels)
#         self.shortcut_relu = nn.ReLU(inplace=True)

#         # Main path
#         # First conv (applies to upsampled input)
#         self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm1d(out_channels)
#         self.relu1 = nn.ReLU(inplace=True)
        
#         # Second conv
#         self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm1d(out_channels)
#         self.relu2 = nn.ReLU(inplace=True)

#     def forward(self, x):
#         # Upsample x first
#         x_upsampled = self.upsample(x)
        
#         # Shortcut path
#         shortcut = self.shortcut_bn(self.shortcut_conv(x_upsampled))

#         # Main path
#         out = self.relu1(self.bn1(self.conv1(x_upsampled)))
#         out = self.bn2(self.conv2(out))
        
#         out += shortcut
#         out = self.relu2(out) # Final activation
#         return out

# class ECGDecoder(nn.Module):
#     """
#     Decoder for ResNet50_1D. Reconstructs ECG from embedding.
#     """
#     def __init__(self, input_embed_dim=OUTPUT_EMBED_DIM, ecg_output_length=INPUT_LENGTH):
#         super(ECGDecoder, self).__init__()
#         self.input_embed_dim = input_embed_dim
#         self.ecg_output_length = ecg_output_length
        
#         # Encoder's final feature map shape before avgpool is (B, 2048, 75)
#         self.start_channels = BASE_WIDTH * 8 * EXPANSION # 2048
#         self.start_length = 75 # This is data-dependent, 2400 -> 1200 -> 600 -> 300 -> 150 -> 75

#         # 1. Reverse FC and AvgPool
#         self.fc_decoder = nn.Linear(self.input_embed_dim, self.start_channels)
#         # We will reshape to (B, 2048, 1) and then upsample to start_length
#         self.un_avgpool = nn.Upsample(size=self.start_length, mode='nearest')
        
#         # Decoder blocks (reversing the encoder layers)
#         # Strides mirror the encoder's downsampling
        
#         # Reverse layer4 (s=2)
#         self.d_layer4 = DecoderBlock(2048, 1024, stride=2) # Out: (B, 1024, 150)
        
#         # Reverse layer3 (s=2)
#         self.d_layer3 = DecoderBlock(1024, 512, stride=2) # Out: (B, 512, 300)
        
#         # Reverse layer2 (s=2)
#         self.d_layer2 = DecoderBlock(512, 256, stride=2) # Out: (B, 256, 600)
        
#         # Reverse layer1 (s=1)
#         self.d_layer1 = DecoderBlock(256, 64, stride=1) # Out: (B, 64, 600)
        
#         # Reverse maxpool (s=2)
#         self.d_maxpool = DecoderBlock(64, 64, stride=2) # Out: (B, 64, 1200)
        
#         # Reverse conv1 (s=2)
#         self.d_conv1 = DecoderBlock(64, 32, stride=2) # Out: (B, 32, 2400)
        
#         # Final convolution to get 1 channel
#         # Mirroring encoder's conv1 kernel_size=7, padding=3
#         self.final_conv = nn.Conv1d(32, 1, kernel_size=7, padding=3)
        
#         # Final activation. Tanh is common for signals normalized to [-1, 1]
#         self.final_activation = nn.Tanh()
        
#     def forward(self, x):
#         # x shape: (B, 128)
        
#         # 1. Reverse FC
#         x = self.fc_decoder(x) # (B, 2048)
        
#         # 2. Reshape and Un-AvgPool
#         x = x.view(x.size(0), self.start_channels, 1) # (B, 2048, 1)
#         x = self.un_avgpool(x) # (B, 2048, 75)
        
#         # 3. Decoder blocks
#         x = self.d_layer4(x) # (B, 1024, 150)
#         x = self.d_layer3(x) # (B, 512, 300)
#         x = self.d_layer2(x) # (B, 256, 600)
#         x = self.d_layer1(x) # (B, 64, 600)
#         x = self.d_maxpool(x) # (B, 64, 1200)
#         x = self.d_conv1(x) # (B, 32, 2400)
        
#         # 4. Final conv
#         x = self.final_conv(x) # (B, 1, 2400)
        
#         # 5. Final activation
#         x = self.final_activation(x)
        
#         return x
class PPGtoECGConverter(nn.Module):
    def __init__(self, embed_dim=OUTPUT_EMBED_DIM, input_length=INPUT_LENGTH):
        super().__init__()
        
        # --- PHẦN 1: PPG ENCODER (LẤY TỪ CLIP) ---
        # Lưu ý: Chúng ta khởi tạo nó y hệt như trong class CLIP
        # Tôi đổi tên nó thành 'ppg_encoder' cho rõ ràng
        self.ppg_encoder = SignalTransformer(
            input_length=input_length,
            patch_size=30,
            width=512,
            layers=6,
            heads=8,
            output_dim=embed_dim
        )
        
        # --- PHẦN 2: ECG DECODER (MỚI THIẾT KẾ) ---
        self.ecg_decoder = ECGDecoder(
            embed_dim=OUTPUT_EMBED_DIM,
            start_channels=512, start_length=75
        )

    def forward(self, ppg_signal):
        # ppg_signal shape: (B, 1, 2400)
        
        # 1. Nén PPG thành vector đặc trưng
        # (B, 1, 2400) -> (B, 128)
        z_ppg = self.ppg_encoder(ppg_signal)
        
        # 2. Giải nén vector thành tín hiệu ECG
        # (B, 128) -> (B, 1, 2400)
        predicted_ecg = self.ecg_decoder(z_ppg)
        
        return predicted_ecg