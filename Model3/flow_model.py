import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, context, t_emb):
        # x: [B, L, D] (ECG tokens)
        # context: [B, L, D] (PPG tokens)
        # t_emb: [B, D] (Thời gian t được cộng/nhúng thêm)
        
        # 1. Self-Attention + Time (Cộng t_emb vào để mạng nhận biết bước thời gian)
        x_norm = self.norm1(x + t_emb.unsqueeze(1))
        x = x + self.self_attn(x_norm, x_norm, x_norm)[0]
        
        # 2. Cross-Attention: x làm Queries, context (PPG) làm Keys/Values
        x_norm = self.norm2(x)
        x = x + self.cross_attn(x_norm, context, context)[0]
        
        # 3. Feed Forward
        x = x + self.mlp(self.norm3(x))
        return x

class LatentTransformerFlow(nn.Module):
    def __init__(self, latent_channels=16, latent_length=75, embed_dim=256, num_heads=8, num_layers=6):
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_length = latent_length
        self.embed_dim = embed_dim

        # 1. Embedders: Biến latent (C, L) thành token sequence (L, D)
        # Dùng Conv1D để giữ cấu trúc thời gian cục bộ (Local temporal structure)
        self.ecg_embedder = nn.Conv1d(latent_channels, embed_dim, kernel_size=3, padding=1)
        self.ppg_embedder = nn.Conv1d(latent_channels, embed_dim, kernel_size=3, padding=1)
        
        # 2. Time Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # 3. Transformer Blocks với Cross-Modal Conditioning
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        
        # 4. Output Head: Biến tokens ngược lại thành latent channels
        self.final_norm = nn.LayerNorm(embed_dim)
        self.final_proj = nn.Conv1d(embed_dim, latent_channels, kernel_size=3, padding=1)

    def forward(self, xt, t, z_ppg):
        """
        xt: [Batch, 16, 75] - Trạng thái nhiễu của ECG
        t: [Batch] - Bước thời gian
        z_ppg: [Batch, 16, 75] - Điều kiện từ PPG
        """
        # A. Chuyển đổi sang dạng chuỗi token [B, D, L] -> [B, L, D]
        x = self.ecg_embedder(xt).transpose(1, 2)
        c = self.ppg_embedder(z_ppg).transpose(1, 2)
        
        # B. Nhúng thời gian
        t_emb = self.time_mlp(t) # [B, D]
        
        # C. Đi qua các khối Transformer
        for block in self.blocks:
            x = block(x, c, t_emb)
            
        # D. Trả về kích thước ban đầu để dự đoán Vector Field [B, C, L]
        x = self.final_norm(x).transpose(1, 2) # [B, D, L]
        v_pred = self.final_proj(x)
        
        return v_pred