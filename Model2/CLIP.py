import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

class LayerNorm1D(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)

class ResidualBlock1D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm1D(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.dwconv(x)
        x = self.norm(x)
        
        x = x.transpose(1, 2) 
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.transpose(1, 2) 
        
        return res + x

class ConvNeXtEncoder(nn.Module):
    def __init__(
        self, 
        in_channels: int = 1, 
        base_dims: Tuple[int, int, int, int] = (64, 128, 256, 512), 
        depths: Tuple[int, int, int, int] = (2, 2, 4, 2)
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_dims[0], kernel_size=4, stride=4),
            LayerNorm1D(base_dims[0])
        )
        
        self.stages = nn.ModuleList()
        in_dims = [base_dims[0]] + list(base_dims[:-1])
        
        for i in range(4):
            downsample = i > 0
            stage = nn.Sequential(
                LayerNorm1D(in_dims[i]) if downsample else nn.Identity(),
                nn.Conv1d(in_dims[i], base_dims[i], kernel_size=2, stride=2) if downsample else nn.Identity(),
                *[ResidualBlock1D(base_dims[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = self.stem(x)
        skips = []
        for stage in self.stages:
            x = stage(x)
            skips.append(x)
        return skips[-1], skips[:-1]

class SpectralEncoder(nn.Module):
    def __init__(
        self, 
        stft_n_fft: int = 128, 
        stft_hop_length: int = 32, 
        fusion_embed_dim: int = 256
    ):
        super().__init__()
        self.n_fft = stft_n_fft
        self.hop_length = stft_hop_length
        
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=(2, 1), padding=1), nn.GELU(),
            nn.Conv2d(64, fusion_embed_dim, 3, padding=1), nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            x.squeeze(1), 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            return_complex=True, 
            center=True
        )
        mag = torch.log1p(spec.abs()).unsqueeze(1) 
        feat = self.conv_net(mag)
        return feat.mean(dim=2).transpose(1, 2) # [B, Time, Channel]


class CrossAttentionFusion(nn.Module):
    def __init__(self, time_dim: int, freq_dim: int, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.q_proj = nn.Linear(time_dim, embed_dim)
        self.kv_proj = nn.Linear(freq_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        
        self.out_proj = nn.Linear(embed_dim, time_dim)
        
        self.norm = nn.LayerNorm(time_dim)

    def forward(self, time_feat: torch.Tensor, freq_feat: torch.Tensor) -> torch.Tensor:
        q = time_feat.transpose(1, 2)             # [B, L, 512]
        query = self.q_proj(q)                    # [B, L, 256]
        key_val = self.kv_proj(freq_feat)         # [B, L_f, 256]

        attn_out, _ = self.attn(query, key_val, key_val) # attn_out: [B, L, 256]
        
        out = self.out_proj(attn_out)             # out: [B, L, 512]
        
        return self.norm(out + q).transpose(1, 2) # Trả về [B, 512, L]

class ECGDecoder(nn.Module):
    def __init__(self, base_dims: Tuple[int, int, int, int] = (64, 128, 256, 512)):
        super().__init__()
        dims = base_dims[::-1] 
        
        self.up_blocks = nn.ModuleList([
            nn.Sequential(
                # nn.Upsample(scale_factor=2, mode='linear'),
                nn.Conv1d(dims[i] + dims[i+1], dims[i+1], 3, padding=1),
                nn.GELU()
            ) for i in range(len(dims)-1)
        ])
        
        self.final_head = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='linear'),
            nn.Conv1d(dims[-1], 1, kernel_size=7, padding=3)
        )

    def forward(self, fused: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        x = fused
        for i, up in enumerate(self.up_blocks):
            skip = skips[-(i+1)]
            x = F.interpolate(x, size=skip.shape[-1], mode='linear')
            x = torch.cat([x, skip], dim=1)
            x = up(x)
        return self.final_head(x)

class PPG2ECGModel(nn.Module):
    def __init__(
        self, 
        in_channels: int = 1,
        base_dims: Tuple[int, int, int, int] = (64, 128, 256, 512),
        depths: Tuple[int, int, int, int] = (2, 2, 4, 2),
        stft_n_fft: int = 128,
        stft_hop_length: int = 32,
        fusion_embed_dim: int = 256,
        proj_dim: int = 128
    ):
        super().__init__()
        
        # 1. Encoders
        self.ppg_time_enc = ConvNeXtEncoder(in_channels, base_dims, depths)
        self.ppg_freq_enc = SpectralEncoder(stft_n_fft, stft_hop_length, fusion_embed_dim)
        
        # 2. Fusion
        self.fusion = CrossAttentionFusion(
            time_dim=base_dims[-1], 
            freq_dim=fusion_embed_dim, 
            embed_dim=fusion_embed_dim
        )
        
        # 3. Decoder
        self.decoder = ECGDecoder(base_dims)
        
        # 4. Contrastive Learning Head 
        self.ecg_enc = ConvNeXtEncoder(in_channels, base_dims, depths)
        self.proj_head = nn.Sequential(nn.Linear(base_dims[-1], proj_dim))

    def forward(self, ppg: torch.Tensor, ecg: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        time_feat, skips = self.ppg_time_enc(ppg)
        freq_feat = self.ppg_freq_enc(ppg)
        
        fused = self.fusion(time_feat, freq_feat)
        recon_ecg = self.decoder(fused, skips)
        
        results = {"recon_ecg": recon_ecg}
        
        if ecg is not None:
            ecg_feat, _ = self.ecg_enc(ecg)
            results["z_ppg"] = self.proj_head(fused.mean(dim=-1))
            results["z_ecg"] = self.proj_head(ecg_feat.mean(dim=-1))
            
        return results