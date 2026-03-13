import torch
from torch import nn
from ResNet50_2 import ResNet50_1D, LayerNorm1d
import torch.nn.functional as F

OUTPUT_EMBED_DIM = 128 
INPUT_LENGTH = 2400
BASE_WIDTH = 64
EXPANSION = 4 
LAYERS = [3, 4, 6, 3] 

# --- MODULE DUNG HỢP MỚI ---
class CrossAttentionFusion(nn.Module):
    def __init__(self, time_channels=2048, freq_channels=256, embed_dim=512, num_heads=8):
        super(CrossAttentionFusion, self).__init__()
        
        self.query_proj = nn.Linear(time_channels, embed_dim)
        self.kv_proj = nn.Linear(freq_channels, embed_dim)
        
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        
        self.output_proj = nn.Linear(embed_dim, time_channels)
        self.layer_norm = nn.LayerNorm(time_channels)

    def forward(self, time_feat, freq_feat):
        # Định dạng lại shape
        Q = time_feat.transpose(1, 2)  # [B, L, 2048]
        Q = self.query_proj(Q)         # [B, L, 512]
        
        K = freq_feat.unsqueeze(1)     # [B, 1, 256] 
        K = self.kv_proj(K)            # [B, 1, 512]
        V = K                          

        # Cross-Attention
        attn_output, _ = self.attention(query=Q, key=K, value=V)
        
        # Residual Connection & Normalization
        out = self.output_proj(attn_output)
        fused_feat = self.layer_norm(out + time_feat.transpose(1, 2))
        
        # Trả về shape [B, Channels, Length]
        return fused_feat.transpose(1, 2)


class DualDomainEncoder(nn.Module):
    def __init__(self, resnet_layers=[3, 4, 6, 3], fft_dim=256):
        super(DualDomainEncoder, self).__init__()
        self.time_branch = ResNet50_1D(layers=resnet_layers)
        self.freq_branch = FFT_MLP(input_length=2400, embed_dim=fft_dim)
        
        # Khởi tạo Cross-Attention Fusion
        self.fusion_module = CrossAttentionFusion(time_channels=2048, freq_channels=fft_dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        f4, features_list = self.time_branch(x) 
        freq_feat = self.freq_branch(x) 
        
        # Sử dụng Cross-Attention thay cho torch.cat
        fused_3d = self.fusion_module(f4, freq_feat) 
        
        fused_1d = self.avgpool(fused_3d).squeeze(-1) 
        return fused_1d, fused_3d, features_list
    

class ECGEssembleCLIP(nn.Module):
    def __init__(self, embed_dim=128, fft_dim=256):
        super(ECGEssembleCLIP, self).__init__()

        # Đã thay đổi: Cross-Attention giữ nguyên số chiều của ResNet là 2048
        self.fused_dim = 2048

        # Encoders
        self.encode_ecg = DualDomainEncoder(fft_dim=fft_dim)
        self.encode_ppg = DualDomainEncoder(fft_dim=fft_dim)

        # Projectors 
        self.project_ecg = nn.Sequential(
            nn.Linear(self.fused_dim, 1024),
            nn.LayerNorm(1024),
            nn.PReLU(),
            nn.Linear(1024, embed_dim)
        )
        
        self.project_ppg = nn.Sequential(
            nn.Linear(self.fused_dim, 1024),
            nn.LayerNorm(1024),
            nn.PReLU(),
            nn.Linear(1024, embed_dim)
        )

        # ECG Converter 
        self.decoder = ECGDecoder_UNet(bottleneck_channels=self.fused_dim, target_length=2400)

    def forward(self, ecg_original, ppg_original):
        ppg_fused_1d, ppg_fused_3d, ppg_features_list = self.encode_ppg(ppg_original)
        
        reconstructed_ecg = self.decoder(ppg_fused_3d, ppg_features_list)

        if ecg_original is None:
            PPG_embedding = self.project_ppg(ppg_fused_1d)
            ppg_norm = PPG_embedding / PPG_embedding.norm(dim=1, keepdim=True)
            return reconstructed_ecg

        else:
            ecg_fused_1d, _, _ = self.encode_ecg(ecg_original)
            
            ECG_embedding = self.project_ecg(ecg_fused_1d)
            PPG_embedding = self.project_ppg(ppg_fused_1d)

            # Chuẩn hóa L2
            ecg_norm = ECG_embedding / ECG_embedding.norm(dim=1, keepdim=True)
            ppg_norm = PPG_embedding / PPG_embedding.norm(dim=1, keepdim=True)
            
            return ecg_norm, ppg_norm, reconstructed_ecg


class DecoderBlock_UNet(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='linear', align_corners=False)
        
        total_in_channels = in_channels + skip_channels
        
        self.conv = nn.Sequential(
            nn.Conv1d(total_in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm1d(out_channels),
            nn.PReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm1d(out_channels),
            nn.PReLU()
        )

    def forward(self, x, skip=None):
        x = self.upsample(x)
        
        if skip is not None:
            if x.size(2) != skip.size(2):
                x = nn.functional.interpolate(x, size=skip.size(2), mode='nearest')
            
            x = torch.cat([x, skip], dim=1) 
            
        return self.conv(x)


class ECGDecoder_UNet(nn.Module):
    # Đã thay đổi: mặc định bottleneck_channels = 2048
    def __init__(self, bottleneck_channels=2048,  target_length=2400):
        super().__init__()
        self.target_length = target_length
        self.adapter = nn.Sequential(
            nn.Conv1d(bottleneck_channels, 512, kernel_size=1),
            LayerNorm1d(512),
            nn.PReLU()
        ) 

        self.block1 = DecoderBlock_UNet(in_channels=512, out_channels=256, skip_channels=1024)
        self.block2 = DecoderBlock_UNet(in_channels=256, out_channels=128, skip_channels=512)
        self.block3 = DecoderBlock_UNet(in_channels=128, out_channels=64, skip_channels=256)
        self.block4 = DecoderBlock_UNet(in_channels=64, out_channels=32, skip_channels=0)
        self.block5 = DecoderBlock_UNet(in_channels=32, out_channels=16, skip_channels=0)

        self.final_conv = nn.Conv1d(16, 1, kernel_size=1)
        self.refine_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, padding=7), 
            LayerNorm1d(16),
            nn.PReLU(),
            nn.Conv1d(16, 1, kernel_size=1) 
        )

    def forward(self, z, features_list):
        f1, f2, f3 = features_list 
        
        x = self.adapter(z) 
        x = self.block1(x, skip=f3) 
        x = self.block2(x, skip=f2) 
        x = self.block3(x, skip=f1) 
        
        x = self.block4(x) 
        x = self.block5(x) 
        
        x = self.final_conv(x)
        if x.shape[-1] != self.target_length:
            x = nn.functional.interpolate(
                x, 
                size=self.target_length, 
                mode='linear', 
                align_corners=False
            )
            x = self.refine_conv(x)
        return x


class FFT_MLP(nn.Module):
    def __init__(self, input_length=2400, embed_dim=256):
        super(FFT_MLP, self).__init__()
        self.fft_len = input_length // 2 + 1 
        
        self.mlp = nn.Sequential(
            nn.Linear(self.fft_len, 512),
            nn.LayerNorm(512),
            nn.PReLU(),
            nn.Linear(512, embed_dim)
        )

    def forward(self, x):
        x_squeeze = x.squeeze(1) 
        fft_x = torch.fft.rfft(x_squeeze)
        mag_x = torch.abs(fft_x) 
        
        freq_feat = self.mlp(mag_x) 
        return freq_feat