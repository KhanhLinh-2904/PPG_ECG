import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from ResNet50 import ResNet50_1D


class LayerNorm1dChannelFirst(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


# ============================================================
# FFT branch: keep multiple spectral tokens instead of collapsing to 1 vector
# ============================================================
class FFTFeatureTokenizer(nn.Module):
 
    def __init__(self, in_channels: int = 2048, fft_bins_keep: int = 64, token_dim: int = 256):
        super().__init__()
        self.fft_bins_keep = fft_bins_keep
        self.proj = nn.Sequential(
            nn.Linear(in_channels, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )

    def forward(self, time_features: torch.Tensor) -> torch.Tensor:
        # time_features: [B, C, L]
        orig_dtype = time_features.dtype
        x = time_features.float()

        fft_out = torch.fft.rfft(x, dim=-1)                # [B, C, Lf]
        mag_out = torch.log1p(torch.abs(fft_out))          # [B, C, Lf]

        # keep first K bins only; shape -> [B, C, K]
        mag_out = mag_out[:, :, :self.fft_bins_keep]

        # turn frequency bins into tokens: [B, K, C]
        freq_tokens = mag_out.transpose(1, 2).contiguous()
        freq_tokens = self.proj(freq_tokens)               # [B, K, token_dim]
        return freq_tokens.to(orig_dtype)


# ============================================================
# Cross-attention fusion with residual and output projection
# ============================================================
class CrossAttentionFusion(nn.Module):
    """
    Query: time tokens   [B, Lt, Ct]
    Key/Val: freq tokens [B, K, Cf]
    Output: fused time feature [B, Ct, Lt]
    """
    def __init__(self, time_channels: int = 2048, freq_channels: int = 256, embed_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.time_in = nn.Linear(time_channels, embed_dim)
        self.freq_in = nn.Linear(freq_channels, embed_dim)

        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        self.out_proj = nn.Linear(embed_dim, time_channels)
        self.out_norm = LayerNorm1dChannelFirst(time_channels)

    def forward(self, time_feat: torch.Tensor, freq_tokens: torch.Tensor) -> torch.Tensor:
        # time_feat: [B, Ct, Lt]
        # freq_tokens: [B, K, Cf]
        time_tokens = time_feat.transpose(1, 2)            # [B, Lt, Ct]
        q = self.time_in(time_tokens)                      # [B, Lt, E]
        kv = self.freq_in(freq_tokens)                     # [B, K, E]

        attn_output, _ = self.attn(query=q, key=kv, value=kv)
        x = self.norm1(q + attn_output)
        x = self.norm2(x + self.ffn(x))

        x = self.out_proj(x)                               # [B, Lt, Ct]
        x = x + time_tokens                                # residual to original time features
        x = x.transpose(1, 2)                              # [B, Ct, Lt]
        x = self.out_norm(x)
        return x


# ============================================================
# Projection head for contrastive learning
# ============================================================
class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LearnableTemperature(nn.Module):
    def __init__(self, init_temp: float = 0.07):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / init_temp)))

    def forward(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0)


# ============================================================
# Dual-domain encoder
# ============================================================
class DualDomainEncoder(nn.Module):
    """
    Uses the original ResNet50_1D backbone, but fixes the frequency branch and fusion design.
    """
    def __init__(
        self, 
        backbone_channels: int = 2048,
        fft_bins_keep: int = 64,
        freq_token_dim: int = 256,
        fusion_embed_dim: int = 256,
        fusion_heads: int = 8,
        embed_dim: int = 128
    ):
        super().__init__()
        self.time_branch = ResNet50_1D()
        self.freq_branch = FFTFeatureTokenizer(
            in_channels=backbone_channels,
            fft_bins_keep=fft_bins_keep,
            token_dim=freq_token_dim,
        )
        self.fusion_module = CrossAttentionFusion(
            time_channels=backbone_channels,
            freq_channels=freq_token_dim,
            embed_dim=fusion_embed_dim,
            num_heads=fusion_heads,
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.projector = ProjectionHead(backbone_channels, embed_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # ResNet50_1D returns f4 and skip list according to the original code base
        time_feat, skips = self.time_branch(x)             # f4 [B, 2048, 75], skips [f1, f2, f3]
        freq_tokens = self.freq_branch(time_feat)          # [B, K, D]
        fused_3d = self.fusion_module(time_feat, freq_tokens)
        pooled_feat = self.avgpool(fused_3d).squeeze(-1)   # [B, 2048]
        z = self.projector(pooled_feat)                    # [B, embed_dim]

        return {
            "time_feat": time_feat,
            "freq_tokens": freq_tokens,
            "fused_3d": fused_3d,
            "pooled_feat": pooled_feat,
            "z": z,
            "skips": skips,
        }


# ============================================================
# ECG decoder with skip connections
# ============================================================
class DecoderUpBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            LayerNorm1dChannelFirst(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            LayerNorm1dChannelFirst(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ECGDecoderTransformer(nn.Module):
    """
    Decoder improvements over the old code:
    - uses skip features from the encoder to keep more morphology detail
    - uses interpolate + conv instead of full transpose-conv stack everywhere
    - still keeps transformer refinement at bottleneck
    """
    def __init__(
        self, 
        bottleneck_channels: int = 2048, 
        d_model: int = 256, 
        nhead: int = 8, 
        num_layers: int = 4, 
        target_length: int = 2400
    ):
        super().__init__()
        self.target_length = target_length

        self.channel_proj = nn.Conv1d(bottleneck_channels, d_model, kernel_size=1)
        self.pos_encoder = PositionalEncoding1D(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Original ResNet50_1D skip channels: f1=256, f2=512, f3=1024, bottleneck=2048
        self.up3 = DecoderUpBlock1D(d_model + 1024, 512)   # 75 -> 150
        self.up2 = DecoderUpBlock1D(512 + 512, 256)        # 150 -> 300
        self.up1 = DecoderUpBlock1D(256 + 256, 128)        # 300 -> 600

        self.up0 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="linear", align_corners=False),  # 600 -> 2400
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            LayerNorm1dChannelFirst(64),
            nn.GELU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            LayerNorm1dChannelFirst(32),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=7, padding=3),
        )

    def forward(self, z: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        f1, f2, f3 = skips

        x = self.channel_proj(z)               # [B, 256, 75]
        x = x.transpose(1, 2)                  # [B, 75, 256]
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.transpose(1, 2)                  # [B, 256, 75]

        x = F.interpolate(x, size=f3.shape[-1], mode="linear", align_corners=False)
        x = torch.cat([x, f3], dim=1)
        x = self.up3(x)

        x = F.interpolate(x, size=f2.shape[-1], mode="linear", align_corners=False)
        x = torch.cat([x, f2], dim=1)
        x = self.up2(x)

        x = F.interpolate(x, size=f1.shape[-1], mode="linear", align_corners=False)
        x = torch.cat([x, f1], dim=1)
        x = self.up1(x)

        ecg_out = self.up0(x)

        if ecg_out.shape[-1] != self.target_length:
            ecg_out = F.interpolate(ecg_out, size=self.target_length, mode="linear", align_corners=False)

        return ecg_out


# ============================================================
# Main architecture
# ============================================================
class ECG_PPG_Fusion_Model(nn.Module):
    def __init__(
        self,
        input_length: int = 2400,
        embed_dim: int = 128,
        backbone_channels: int = 2048,
        fft_bins_keep: int = 64,
        freq_token_dim: int = 256,
        fusion_embed_dim: int = 256,
        fusion_heads: int = 8,
        decoder_dim: int = 256,
        decoder_layers: int = 4,
        target_length: int = 2400,
        temperature_init: float = 0.07
    ):
        super().__init__()
        
        # Dual-domain Encoders
        self.encode_ecg = DualDomainEncoder(
            backbone_channels=backbone_channels,
            fft_bins_keep=fft_bins_keep,
            freq_token_dim=freq_token_dim,
            fusion_embed_dim=fusion_embed_dim,
            fusion_heads=fusion_heads,
            embed_dim=embed_dim
        )
        
        self.encode_ppg = DualDomainEncoder(
            backbone_channels=backbone_channels,
            fft_bins_keep=fft_bins_keep,
            freq_token_dim=freq_token_dim,
            fusion_embed_dim=fusion_embed_dim,
            fusion_heads=fusion_heads,
            embed_dim=embed_dim
        )
        
        # Transformer Decoder
        self.decoder = ECGDecoderTransformer(
            bottleneck_channels=backbone_channels,
            d_model=decoder_dim,
            nhead=fusion_heads,
            num_layers=decoder_layers,
            target_length=target_length,
        )
        
        # Learnable Temperature
        self.temperature = LearnableTemperature(temperature_init)

    def forward(self, ecg_signal: Optional[torch.Tensor], ppg_signal: torch.Tensor) -> Dict[str, torch.Tensor]:
        # PPG branch: always used
        ppg_out = self.encode_ppg(ppg_signal)
        reconstructed_ecg = self.decoder(ppg_out["fused_3d"], ppg_out["skips"])

        outputs = {
            "reconstructed_ecg": reconstructed_ecg,
            "ppg_z": ppg_out["z"],
            "ppg_pooled_feat": ppg_out["pooled_feat"],
        }

        # ECG branch: only needed during training
        if ecg_signal is not None:
            ecg_out = self.encode_ecg(ecg_signal)
            outputs["ecg_z"] = ecg_out["z"]
            outputs["ecg_pooled_feat"] = ecg_out["pooled_feat"]

        return outputs