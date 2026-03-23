import torch
from torch import nn
from ResNet50 import ResNet50_1D, LayerNorm1d
import torch.nn.functional as F
import math
OUTPUT_EMBED_DIM = 128 
INPUT_LENGTH = 2400
BASE_WIDTH = 64
EXPANSION = 4 
LAYERS = [3, 4, 6, 3] 

class CrossAttentionFusion(nn.Module):
    def __init__(self, time_channels=2048, freq_channels=256, embed_dim=512, num_heads=8):
        super(CrossAttentionFusion, self).__init__()
        
        self.query_proj = nn.Linear(time_channels, embed_dim)
        self.kv_proj = nn.Linear(freq_channels, embed_dim)
        
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        
        self.output_proj = nn.Linear(embed_dim, time_channels)
        self.layer_norm = nn.LayerNorm(time_channels)

    def forward(self, time_feat, freq_feat):
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
        
        # shape [B, Channels, Length]
        return fused_feat.transpose(1, 2)


class DualDomainEncoder(nn.Module):
    def __init__(self, resnet_layers=[3, 4, 6, 3], fft_dim=256):
        super(DualDomainEncoder, self).__init__()
        self.time_branch = ResNet50_1D(layers=resnet_layers)
        self.freq_branch = FFT_MLP(input_length=2400, embed_dim=fft_dim)
        
        # Cross-Attention Fusion
        self.fusion_module = CrossAttentionFusion(time_channels=2048, freq_channels=fft_dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        f4, features_list = self.time_branch(x) 
        freq_feat = self.freq_branch(x) 
        
        # Cross-Attention 
        fused_3d = self.fusion_module(f4, freq_feat) 
        print("shape of fused_3d: ", fused_3d.shape)
        fused_1d = self.avgpool(fused_3d).squeeze(-1) 
        print("shape of fused_1d: ", fused_1d.shape)
        return fused_1d, fused_3d, features_list
    

class ECGEssembleCLIP(nn.Module):
    def __init__(self, embed_dim=128, fft_dim=256):
        super(ECGEssembleCLIP, self).__init__()

        self.fused_dim = 2048

        # Encoders
        self.encode_ecg = DualDomainEncoder(fft_dim=fft_dim)
        self.encode_ppg = DualDomainEncoder(fft_dim=fft_dim)

        
        # ECG Converter 
        self.decoder = ECGDecoder_Transformer(
            bottleneck_channels=self.fused_dim, 
            d_model=256, 
            nhead=8, 
            num_layers=4, 
            target_length=2400
        )

    def forward(self, ecg_original, ppg_original):
        ppg_fused_1d, ppg_fused_3d, ppg_features_list = self.encode_ppg(ppg_original)
        
        reconstructed_ecg = self.decoder(ppg_fused_3d, features_list=None)

        if ecg_original is None:
            # PPG_embedding = self.project_ppg(ppg_fused_1d)
            PPG_embedding = ppg_fused_1d
            ppg_norm = PPG_embedding / PPG_embedding.norm(dim=1, keepdim=True)
            return reconstructed_ecg

        else:
            ecg_fused_1d, ecg_fused_3d, ecg_features_list = self.encode_ecg(ecg_original)
            
            # ECG_embedding = self.project_ecg(ecg_fused_1d)
            # PPG_embedding = self.project_ppg(ppg_fused_1d)
            ECG_embedding = ecg_fused_1d
            PPG_embedding = ppg_fused_1d

            ecg_norm = ECG_embedding / ECG_embedding.norm(dim=1, keepdim=True)
            ppg_norm = PPG_embedding / PPG_embedding.norm(dim=1, keepdim=True)
            
            return ecg_norm, ppg_norm, reconstructed_ecg



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
        # x shape: [Batch, Length, Channels]
        return x + self.pe[:, :x.size(1), :]


class ECGDecoder_Transformer(nn.Module):
    def __init__(self, bottleneck_channels=2048, d_model=256, nhead=8, num_layers=4, target_length=2400):
        super().__init__()
        self.target_length = target_length
        
        # 1. Projector: Down from 2048 to 256 
        self.channel_proj = nn.Conv1d(bottleneck_channels, d_model, kernel_size=1)
        
        # 2. Transformer (Phase Alignment)
        self.pos_encoder = PositionalEncoding1D(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=0.1, 
            activation='gelu', 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Upsampler 
        # L=75 (2400 / 32).
        # 75 -> 150 -> 300 -> 600 -> 1200 -> 2400.
        self.upsampler = nn.Sequential(
            #1: L -> 2L
            nn.ConvTranspose1d(d_model, 128, kernel_size=2, stride=2),
            nn.BatchNorm1d(128), 
            nn.GELU(),
   
            #2: 2L -> 4L
            nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm1d(64), 
            nn.GELU(),
         

            
            #3: 4L -> 8L
            nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm1d(32), 
            nn.GELU(),
           
            #4: 8L -> 16L
            nn.ConvTranspose1d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm1d(16), 
            nn.GELU(),
          
            
            #5: 16L -> 32L (2400)
            nn.ConvTranspose1d(16, 8, kernel_size=2, stride=2),
            nn.BatchNorm1d(8), 
            nn.GELU(),
        
            # Kernel_size=7 
            nn.Conv1d(8, 1, kernel_size=7, padding=3)
        )

    def forward(self, z, features_list=None):
       
        x = self.channel_proj(z)             # [B, 256, L]
        x = x.transpose(1, 2)                # Transformer need shape [B, L, 256]
        
        # 2. Transformer 
        x = self.pos_encoder(x)
        x = self.transformer(x)              # [B, L, 256]
        
        # 3.  CNN and upsampling
        x = x.transpose(1, 2)                # [B, 256, L]
        ecg_out = self.upsampler(x)          # [B, 1, 2400]
        
        if ecg_out.shape[-1] != self.target_length:
            ecg_out = torch.nn.functional.interpolate(ecg_out, size=self.target_length, mode='linear')
            
        return ecg_out

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
        
        current_len = mag_x.shape[-1]
        if current_len < self.fft_len:
            pad_size = self.fft_len - current_len
            mag_x = F.pad(mag_x, (0, pad_size), mode='constant', value=0.0)

        freq_feat = self.mlp(mag_x) 
        return freq_feat
    

if __name__ == "__main__":
    BATCH_SIZE = 32
    CHANNELS = 1
    SEQ_LENGTH = 2400
    EMBED_DIM = 256

    dummy_ecg = torch.randn(BATCH_SIZE, CHANNELS, SEQ_LENGTH)
    model = DualDomainEncoder()
    output_features = model(dummy_ecg)