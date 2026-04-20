import torch
from torch import nn
from ResNet50 import ResNet50_1D, LayerNorm1d
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
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

class BiDirectionalCrossAttentionFusion(nn.Module):
    def __init__(self, time_channels=2048, freq_channels=256, embed_dim=512, num_heads=8):
        super(BiDirectionalCrossAttentionFusion, self).__init__()
        
        self.time_proj = nn.Linear(time_channels, embed_dim)
        self.freq_proj = nn.Linear(freq_channels, embed_dim)
        
        # 1: Time -> Frequency
        self.attn_T_to_F = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        # 2: Frequency -> Time
        self.attn_F_to_T = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        
        # 3. Projection 
        self.output_proj = nn.Linear(embed_dim * 2, time_channels)
        self.layer_norm = nn.LayerNorm(time_channels)
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, time_feat, freq_feat):
        # time_feat: [B, 2048, L] -> [B, L, 2048]
        T_orig = time_feat.transpose(1, 2)  
        T = self.time_proj(T_orig)          # [B, L, 512]
        
        # freq_feat: [B, 256] -> [B, 1, 256]
        F_orig = freq_feat.unsqueeze(1)     
        F = self.freq_proj(F_orig)          # [B, 1, 512]

        
        out_T2F, _ = self.attn_T_to_F(query=T, key=F, value=F)
        out_F2T, _ = self.attn_F_to_T(query=F, key=T, value=T)
        
        # --- FUSION --
        L = T.size(1)
        out_F2T_expanded = out_F2T.expand(-1, L, -1)
        
        #(Concatenation) -> [B, L, 1024]
        fused_rep = torch.cat([out_T2F, out_F2T_expanded], dim=-1)
        
        # --- RESIDUAL & NORMALIZATION ---
        out = self.output_proj(fused_rep) # [B, L, 2048]
        gate = torch.tanh(self.gamma)
        fused_feat = self.layer_norm(T_orig + gate * out) # [B, L, 2048]
        
        #  [B, Channels, Length]
        return fused_feat.transpose(1, 2)
class DualDomainEncoder(nn.Module):
    def __init__(self, resnet_layers=[3, 4, 6, 3], fft_dim=256, use_projector=False, fused_dim=2048, ecg_branch=False):
        super(DualDomainEncoder, self).__init__()
        self.time_branch = ResNet50_1D(layers=resnet_layers)
        if ecg_branch:
            self.freq_branch = FFT_MLP(input_length=2400, embed_dim=fft_dim)
        else:
            self.freq_branch = FFT_MLP(input_length=2400, embed_dim=fft_dim)
        
        # Cross-Attention Fusion
        self.fusion_module = CrossAttentionFusion(time_channels=2048, freq_channels=fft_dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.use_projector = use_projector
        if self.use_projector:
            self.projector = PPGProjector(channels=fused_dim, hidden_dim=512)
    def forward(self, x):
        f4, features_list = self.time_branch(x) 
        freq_feat = self.freq_branch(x) 
        # Testing when FFT-MLP is 0
        # freq_feat = torch.zeros_like(freq_feat)
        # Cross-Attention 
        fused_3d = self.fusion_module(f4, freq_feat) 
        fused_1d = self.avgpool(fused_3d).squeeze(-1) 
        if self.use_projector:
            fused_3d = self.projector(fused_3d)
        return fused_1d, fused_3d, features_list
    

class PPGProjector(nn.Module):
    def __init__(self, channels=2048, hidden_dim=512):
        super(PPGProjector, self).__init__()
        self.net = nn.Sequential(
        nn.Conv1d(channels, hidden_dim, kernel_size=1),
        nn.BatchNorm1d(hidden_dim),
        nn.GELU(),
        nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        nn.BatchNorm1d(hidden_dim),
        nn.GELU(),
        nn.Conv1d(hidden_dim, channels, kernel_size=1),
        nn.BatchNorm1d(channels)) 
    def forward(self, x):
        return x + self.net(x) 
class ECGEssembleCLIP(nn.Module):
    def __init__(self, embed_dim=128, fft_dim=256):
        super(ECGEssembleCLIP, self).__init__()

        self.fused_dim = 2048

        # Encoders
        self.encode_ecg = DualDomainEncoder(fft_dim=fft_dim, ecg_branch=True)
        self.encode_ppg = DualDomainEncoder(fft_dim=fft_dim, ecg_branch=False)

        
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
        ecg_fused_1d, ecg_fused_3d, ecg_features_list = self.encode_ecg(ecg_original)
       
        # sim = F.cosine_similarity(ecg_fused_1d, ppg_fused_1d, dim=1).mean()
        # print("similarity: ", sim)
        reconstructed_ecg = self.decoder(ppg_fused_3d, features_list=None)

        if ecg_original is None:
            PPG_embedding = ppg_fused_1d
            ppg_norm = PPG_embedding / PPG_embedding.norm(dim=1, keepdim=True)
            return reconstructed_ecg

        else:
            ecg_fused_1d, ecg_fused_3d, ecg_features_list = self.encode_ecg(ecg_original)
        
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

class FFT_MLP_low(nn.Module):
    def __init__(self, input_length=2400, embed_dim=256, order=4):
        super().__init__()

        self.draw = True

        self.fft_len = input_length // 2 + 1
        self.embed_dim = embed_dim
        self.order = order
        
        freqs = torch.linspace(0, 1, self.fft_len)
        self.register_buffer('freqs', freqs)
        
        self.fc = nn.Parameter(torch.tensor([0.1])) 

    def forward(self, x):
        fs = 125
        x_squeeze = x.squeeze(1) 
        fft_x = torch.fft.rfft(x_squeeze)
        mag_x = torch.abs(fft_x)
        
        current_len = mag_x.shape[-1]
        if current_len < self.fft_len:
            pad_size = self.fft_len - current_len
            mag_x = F.pad(mag_x, (0, pad_size), mode='constant', value=0.0)

        fc_safe = torch.clamp(self.fc, min=1e-3) 
        freqs_safe = torch.clamp(self.freqs, min=1e-5)
        
        H = 1.0 / (1.0 + ( freqs_safe / fc_safe ) ** (2 * self.order))
        
        filtered_mag = mag_x * H 
        
        freq_feat = filtered_mag[:, :self.embed_dim]

        #### draw ############
        # print("Low pass filter: ", self.draw)
        # if self.draw: 
        #     mag_before = mag_x[0].detach().cpu().numpy()
        #     mag_after = filtered_mag[0].detach().cpu().numpy()
        #     filter_curve = H.detach().cpu().numpy()
            
        
        #     freqs_np = self.freqs.cpu().numpy() * (fs / 2.0)
        #     fc_np = fc_safe.item() * (fs / 2.0)

        #     plt.figure(figsize=(12, 6))

        #     plt.subplot(2, 1, 1)
        #     plt.plot(freqs_np, mag_before, color='gray', label='Original Magnitude')
            
        #     max_mag = mag_before.max() if mag_before.max() > 0 else 1
        #     plt.plot(freqs_np, filter_curve * max_mag, color='red', linestyle='--', alpha=0.7, label=f'Butterworth Filter H (Order={self.order})')
        #     plt.axvline(x=fc_np, color='orange', linestyle=':', linewidth=2, label=f'Learned Cutoff (fc ≈ {fc_np:.2f} Hz)')
            
        #     plt.title("Phổ tần số TRƯỚC khi lọc (Vạch cam là tần số cắt mạng tự học)")
        #     plt.ylabel("Biên độ")
        #     plt.xlim(0, 10)  
        #     plt.legend(loc="upper right")
        #     plt.grid(True, alpha=0.3)

        #     plt.subplot(2, 1, 2)
        #     plt.plot(freqs_np, mag_after, color='green', label='Filtered Magnitude')
        #     plt.axvline(x=fc_np, color='orange', linestyle=':', linewidth=2)
            
        #     plt.title("Phổ tần số SAU khi lọc Low-Pass")
        #     plt.xlabel("Tần số (Hz)")
        #     plt.ylabel("Biên độ")
        #     plt.xlim(0, 10)
        #     plt.legend(loc="upper right")
        #     plt.grid(True, alpha=0.3)

        #     plt.tight_layout()
        #     plt.show()
        #     self.draw = False
        #### draw ############
        return freq_feat
    
class FFT_MLP_high(nn.Module):
    def __init__(self, input_length=2400, embed_dim=256, order=4):
        super().__init__()
        self.draw = True
        self.fft_len = input_length // 2 + 1
        self.embed_dim = embed_dim
        self.order = order
        
        freqs = torch.linspace(0, 1, self.fft_len)
        self.register_buffer('freqs', freqs)
        
        self.fc = nn.Parameter(torch.tensor([0.1])) 

    def forward(self, x):
        fs = 125
        x_squeeze = x.squeeze(1) 
        fft_x = torch.fft.rfft(x_squeeze)
        mag_x = torch.abs(fft_x)
        
        current_len = mag_x.shape[-1]
        if current_len < self.fft_len:
            pad_size = self.fft_len - current_len
            mag_x = F.pad(mag_x, (0, pad_size), mode='constant', value=0.0)

        fc_safe = torch.clamp(self.fc, min=1e-3) 
        freqs_safe = torch.clamp(self.freqs, min=1e-5) 
        
        H = 1.0 / (1.0 + (fc_safe / freqs_safe) ** (2 * self.order))
        
        filtered_mag = mag_x * H 
        
        freq_feat = filtered_mag[:, -self.embed_dim:]
        #### draw ############
        # print("High pass filter: ", self.draw)

        # if self.draw:
        #     mag_before = mag_x[0].detach().cpu().numpy()
        #     mag_after = filtered_mag[0].detach().cpu().numpy()
        #     filter_curve = H.detach().cpu().numpy()
            
        
        #     freqs_np = self.freqs.cpu().numpy() * (fs / 2.0)
        #     fc_np = fc_safe.item() * (fs / 2.0)

        #     plt.figure(figsize=(12, 6))

        #     plt.subplot(2, 1, 1)
        #     plt.plot(freqs_np, mag_before, color='gray', label='Original Magnitude')
            
        #     max_mag = mag_before.max() if mag_before.max() > 0 else 1
        #     plt.plot(freqs_np, filter_curve * max_mag, color='red', linestyle='--', alpha=0.7, label=f'Butterworth Filter H (Order={self.order})')
        #     plt.axvline(x=fc_np, color='orange', linestyle=':', linewidth=2, label=f'Learned Cutoff (fc ≈ {fc_np:.2f} Hz)')
            
        #     plt.title("Phổ tần số TRƯỚC khi lọc (Vạch cam là tần số cắt mạng tự học)")
        #     plt.ylabel("Biên độ")
        #     plt.xlim(0, 10)  
        #     plt.legend(loc="upper right")
        #     plt.grid(True, alpha=0.3)

        #     plt.subplot(2, 1, 2)
        #     plt.plot(freqs_np, mag_after, color='green', label='Filtered Magnitude')
        #     plt.axvline(x=fc_np, color='orange', linestyle=':', linewidth=2)
            
        #     plt.title("Phổ tần số SAU khi lọc High-Pass")
        #     plt.xlabel("Tần số (Hz)")
        #     plt.ylabel("Biên độ")
        #     plt.xlim(0, 10)
        #     plt.legend(loc="upper right")
        #     plt.grid(True, alpha=0.3)

        #     plt.tight_layout()
        #     plt.show()
        #     self.draw = False
        #### draw ############
        return freq_feat
    
    
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