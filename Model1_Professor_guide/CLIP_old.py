import torch
from torch import nn
from ResNet50 import ResNet50_1D
import torch.nn.functional as F
import math
OUTPUT_EMBED_DIM = 128 
INPUT_LENGTH = 2400
BASE_WIDTH = 64
EXPANSION = 4 
LAYERS = [3, 4, 6, 3] 

class FFT_FeatureExtractor(nn.Module):
    def __init__(self, in_channels=2048, freq_channels=2048):
        super(FFT_FeatureExtractor, self).__init__()

    def forward(self, time_features):
        orig_dtype = time_features.dtype
        time_features_f32 = time_features.to(torch.float32)
        fft_out = torch.fft.rfft(time_features_f32, dim=-1)
        mag_out = torch.abs(fft_out)  
        mag_out = mag_out.to(orig_dtype)
        freq_pooled = mag_out.mean(dim=-1) 
        
        return freq_pooled
class CrossAttentionFusion(nn.Module):
    def __init__(self, time_channels=2048, freq_channels=2048, embed_dim=512, num_heads=8):
        super(CrossAttentionFusion, self).__init__()
        self.query_proj = nn.Linear(time_channels, embed_dim)
        self.kv_proj = nn.Linear(freq_channels, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        
        # self.out_proj = nn.Linear(embed_dim, time_channels)
        
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, time_feat, freq_feat):
        Q_orig = time_feat.transpose(1, 2)  
        Q_proj = self.query_proj(Q_orig)         # [B, L, 512]
        
        # freq_feat: [B, 256] -> [B, 1, 256]
        K = freq_feat.unsqueeze(1)     
        K_proj = self.kv_proj(K)            # [B, 1, 512]
        V_proj = K_proj                          
        
        # Cross-Attention output: [B, L, 512]
        attn_output, _ = self.attention(query=Q_proj, key=K_proj, value=V_proj)
        
        # attn_output_projected = self.out_proj(attn_output)  # [B, L, 2048]
        
        fused_feat = self.layer_norm( attn_output)
        return fused_feat.transpose(1, 2)

class DualDomainEncoder(nn.Module):
    def __init__(self, resnet_layers=[3, 4, 6, 3], in_channels=2048, freq_dim=2048):
        super(DualDomainEncoder, self).__init__()
        self.time_branch = ResNet50_1D(layers=resnet_layers)
        self.freq_branch = FFT_FeatureExtractor(in_channels=in_channels, freq_channels=freq_dim)
        
        self.fusion_module = CrossAttentionFusion(time_channels=in_channels, freq_channels=freq_dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
      
    def forward(self, x):
        time_feat, features_list = self.time_branch(x) 
        freq_feat = self.freq_branch(time_feat) 
       
        fused_3d = self.fusion_module(time_feat, freq_feat) 
        fused_1d = self.avgpool(fused_3d).squeeze(-1) 

        return fused_1d, fused_3d

class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) 

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class ECGDecoder_Transformer(nn.Module):
    def __init__(self, bottleneck_channels=512, d_model=256, nhead=8, num_layers=4, target_length=2400):
        super().__init__()
        self.target_length = target_length
        self.channel_proj = nn.Conv1d(bottleneck_channels, d_model, kernel_size=1)
        self.pos_encoder = PositionalEncoding1D(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.upsampler = nn.Sequential(
            nn.ConvTranspose1d(d_model, 128, kernel_size=2, stride=2),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.ConvTranspose1d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm1d(16), nn.GELU(),
            nn.ConvTranspose1d(16, 8, kernel_size=2, stride=2),
            nn.BatchNorm1d(8), nn.GELU(),
            nn.Conv1d(8, 1, kernel_size=7, padding=3)
        )

    def forward(self, z):
        x = self.channel_proj(z)             
        x = x.transpose(1, 2)                
        x = self.pos_encoder(x)
        x = self.transformer(x)              
        x = x.transpose(1, 2)                
        ecg_out = self.upsampler(x)          
        
        if ecg_out.shape[-1] != self.target_length:
            ecg_out = torch.nn.functional.interpolate(ecg_out, size=self.target_length, mode='linear')
            
        return ecg_out


# ==========================================
# 3. MAIN ARCHITECTURE
# ==========================================
class ECG_PPG_Fusion_Model(nn.Module):
    def __init__(self, embed_dim=128, freq_dim=2048, target_length=2400):
        super(ECG_PPG_Fusion_Model, self).__init__()
        self.fused_dim = 2048 

        self.encode_ecg = DualDomainEncoder(in_channels=self.fused_dim, freq_dim=self.fused_dim)
        self.encode_ppg = DualDomainEncoder(in_channels=self.fused_dim, freq_dim=self.fused_dim)

        self.decoder = ECGDecoder_Transformer(
            bottleneck_channels=512, 
            d_model=256, 
            nhead=8, 
            num_layers=4, 
            target_length=target_length
        )

    def forward(self, ecg_signal, ppg_signal):
        z_ppg_1d, z_ppg_3d = self.encode_ppg(ppg_signal)
        reconstructed_ecg = self.decoder(z_ppg_3d)

        if ecg_signal is not None:
            z_ecg_1d, z_ecg_3d = self.encode_ecg(ecg_signal)
            return z_ecg_1d, z_ppg_1d, reconstructed_ecg

        return reconstructed_ecg