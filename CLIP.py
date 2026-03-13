import torch
from torch import nn
from ResNet50 import ResNet50_1D, LayerNorm1d
import torch.nn.functional as F

OUTPUT_EMBED_DIM = 128 
INPUT_LENGTH = 2400
BASE_WIDTH = 64
EXPANSION = 4 
LAYERS = [3, 4, 6, 3] 
class CrossAttention1D(nn.Module):
    def __init__(self, query_dim, key_dim, hidden_dim):
        super().__init__()
        # Project vector vào không gian Attention
        self.q_proj = nn.Conv1d(query_dim, hidden_dim, 1)
        self.k_proj = nn.Conv1d(key_dim, hidden_dim, 1)
        # Value được map về cùng số kênh với Query để dễ dàng concat
        self.v_proj = nn.Conv1d(key_dim, query_dim, 1) 
        self.scale = hidden_dim ** -0.5

    def forward(self, query, key_value):
        # query: [Batch, Channels_q, Length_q] (Từ Decoder)
        # key_value: [Batch, Channels_k, Length_k] (Từ Skip Connection của PPG)
        
        Q = self.q_proj(query).transpose(1, 2)      # [B, L_q, H]
        K = self.k_proj(key_value)                  # [B, H, L_k]
        V = self.v_proj(key_value).transpose(1, 2)  # [B, L_k, C_q]

        # Tính ma trận Attention (Căn chỉnh thời gian)
        # attn shape: [B, L_q, L_k] - Bản đồ chỉ ra thời điểm t_q của ECG tương ứng với t_k nào của PPG
        attn = torch.bmm(Q, K) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Lấy thông tin PPG đã được dịch pha (Phase-shifted PPG features)
        out = torch.bmm(attn, V).transpose(1, 2)    # [B, C_q, L_q]
        return out
    
class AttentionDecoderBlock1D(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_attention=True):
        super().__init__()
        # ConvTranspose1d giúp tăng gấp đôi chiều dài một cách toán học, thay vì nội suy tuyến tính
        self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
        self.use_attention = use_attention
        if use_attention and skip_channels > 0:
            self.attn = CrossAttention1D(query_dim=in_channels // 2, key_dim=skip_channels, hidden_dim=in_channels // 2)
            conv_in = (in_channels // 2) * 2
        elif skip_channels > 0:
            conv_in = (in_channels // 2) + skip_channels
        else:
            conv_in = in_channels // 2

        # Lớp tinh chỉnh sau khi đã ghép nối
        self.conv = nn.Sequential(
            nn.Conv1d(conv_in, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels), # Bạn có thể dùng LayerNorm1d tùy ý
            nn.PReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.PReLU()
        )

    def forward(self, x, skip=None):
        x = self.up(x) # Tăng độ phân giải thời gian
        
        if skip is not None:
            if self.use_attention:
                # Phép màu xảy ra ở đây: skip connection tự động trượt dọc theo trục thời gian 
                # để khớp pha với x trước khi được nối vào.
                skip = self.attn(query=x, key_value=skip)
            
            # Xử lý chênh lệch kích thước do sai số làm tròn của lớp Pooling ở Encoder
            if x.size(2) != skip.size(2):
                diff = skip.size(2) - x.size(2)
                x = F.pad(x, [diff // 2, diff - diff // 2])
            
            x = torch.cat([x, skip], dim=1)
            
        return self.conv(x)
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
        
        fused_1d = self.avgpool(fused_3d).squeeze(-1) 
        return fused_1d, fused_3d, features_list
    

class ECGEssembleCLIP(nn.Module):
    def __init__(self, embed_dim=128, fft_dim=256):
        super(ECGEssembleCLIP, self).__init__()

        self.fused_dim = 2048

        # Encoders
        self.encode_ecg = DualDomainEncoder(fft_dim=fft_dim)
        self.encode_ppg = DualDomainEncoder(fft_dim=fft_dim)

        # # Projectors 
        # self.project_ecg = nn.Sequential(
        #     nn.Linear(self.fused_dim, 1024),
        #     nn.LayerNorm(1024),
        #     nn.PReLU(),
        #     nn.Linear(1024, embed_dim)
        # )
        
        # self.project_ppg = nn.Sequential(
        #     nn.Linear(self.fused_dim, 1024),
        #     nn.LayerNorm(1024),
        #     nn.PReLU(),
        #     nn.Linear(1024, embed_dim)
        # )

        # ECG Converter 
        self.decoder = ECGDecoder_UNet(bottleneck_channels=self.fused_dim, target_length=2400)

    def forward(self, ecg_original, ppg_original):
        ppg_fused_1d, ppg_fused_3d, ppg_features_list = self.encode_ppg(ppg_original)
        
        reconstructed_ecg = self.decoder(ppg_fused_3d, ppg_features_list)

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
    def __init__(self, bottleneck_channels=2048, target_length=2400):
        super().__init__()
        self.target_length = target_length
        
        # 1. Adapter giảm chiều từ 2048 xuống 512
        self.adapter = nn.Sequential(
            nn.Conv1d(bottleneck_channels, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.PReLU()
        ) 

        # 2. Khai báo các khối Attention Decoder
        # Lưu ý: Các tham số skip_channels phải khớp với số kênh f3, f2, f1 từ ResNet của bạn
        self.block1 = AttentionDecoderBlock1D(in_channels=512, skip_channels=1024, out_channels=256)
        self.block2 = AttentionDecoderBlock1D(in_channels=256, skip_channels=512,  out_channels=128)
        self.block3 = AttentionDecoderBlock1D(in_channels=128, skip_channels=256,  out_channels=64)
        
        # Ở các block cuối không có skip connection, chỉ upsample để đạt đủ 2400 length
        self.block4 = AttentionDecoderBlock1D(in_channels=64, skip_channels=0, out_channels=32, use_attention=False)
        self.block5 = AttentionDecoderBlock1D(in_channels=32, skip_channels=0, out_channels=16, use_attention=False)

        # 3. Chốt chặn cuối cùng sinh ra 1 kênh ECG duy nhất
        self.final_conv = nn.Conv1d(16, 1, kernel_size=1)

    def forward(self, z, features_list):
        # z: ppg_fused_3d [B, 2048, L/32]
        # features_list: [f1, f2, f3] tương ứng với các độ phân giải cao dần
        f1, f2, f3 = features_list 
        
        x = self.adapter(z) 
        x = self.block1(x, skip=f3) 
        x = self.block2(x, skip=f2) 
        x = self.block3(x, skip=f1) 
        
        x = self.block4(x) 
        x = self.block5(x) 
        
        x = self.final_conv(x)
        
        # Output lúc này nhờ ConvTranspose1d đã được đảm bảo kích thước chẵn theo lũy thừa 2.
        # Nếu chiều dài đầu vào gốc của bạn là 2400, sau 5 lần giảm (2^5 = 32), ở đáy là 75.
        # Khi upsample 5 lần, nó sẽ tự động về chẵn 2400 (75 * 32 = 2400) mà KHÔNG cần interpolate!
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
        
        current_len = mag_x.shape[-1]
        if current_len < self.fft_len:
            pad_size = self.fft_len - current_len
            # F.pad format: (pad_left, pad_right). Ta chỉ thêm vào đuôi (cao tần)
            mag_x = F.pad(mag_x, (0, pad_size), mode='constant', value=0.0)

        freq_feat = self.mlp(mag_x) 
        return freq_feat