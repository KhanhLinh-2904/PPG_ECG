import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Config
# ============================================================
@dataclass
class ECGAEConfig:
    input_length: int = 2400
    in_channels: int = 1

    # backbone
    dims: Tuple[int, int, int, int] = (64, 128, 256, 512)
    depths: Tuple[int, int, int, int] = (2, 2, 4, 2)

    # temporal latent
    latent_channels: int = 16         # latent sequence channels
    latent_length: int = 75           # keep temporal structure

    # attention
    attn_heads: int = 8
    attn_dropout: float = 0.0

    # decoder global branch
    global_latent_dim: int = 128
    trend_poly: int = 2


# ============================================================
# Basic modules
# ============================================================
class LayerNorm1D(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class GRN1D(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=2, keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtV2Block1D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Conv1d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.grn = GRN1D(4 * dim)
        self.pwconv2 = nn.Conv1d(4 * dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x


class ResidualConvBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 15):
        super().__init__()
        padding = kernel_size // 2
        self.norm1 = nn.GroupNorm(32 if in_channels >= 32 else 1, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

        self.norm2 = nn.GroupNorm(32 if out_channels >= 32 else 1, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)

        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = F.silu(x)
        x = self.conv2(x)
        return x + residual


class DownsampleBlock1D(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.block = nn.Sequential(
            LayerNorm1D(in_dim),
            nn.Conv1d(in_dim, out_dim, kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpsampleBlock1D(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(in_dim, out_dim, kernel_size=3, padding=1),
            LayerNorm1D(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SelfAttention1D(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        x_seq = x.transpose(1, 2)  # [B, L, C]

        x_norm = self.norm1(x_seq)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_seq = x_seq + attn_out

        x_ffn = self.ffn(self.norm2(x_seq))
        x_seq = x_seq + x_ffn

        return x_seq.transpose(1, 2)


# ============================================================
# Decomposable global branch
# ============================================================
class TrendLayer(nn.Module):
    def __init__(self, seq_len: int, feat_dim: int, latent_dim: int, trend_poly: int):
        super().__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.trend_poly = trend_poly
        self.fc1 = nn.Linear(latent_dim, feat_dim * trend_poly)
        self.fc2 = nn.Linear(feat_dim * trend_poly, feat_dim * trend_poly)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, latent_dim]
        params = F.gelu(self.fc1(z))
        params = self.fc2(params).view(-1, self.feat_dim, self.trend_poly)

        t = torch.arange(0, self.seq_len, device=z.device).float() / self.seq_len
        poly_basis = torch.stack([t ** (p + 1) for p in range(self.trend_poly)], dim=0)  # [P, L]

        out = torch.matmul(params, poly_basis)   # [B, feat_dim, L]
        return out


class LevelLayer(nn.Module):
    def __init__(self, seq_len: int, feat_dim: int, latent_dim: int):
        super().__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.fc1 = nn.Linear(latent_dim, feat_dim)
        self.fc2 = nn.Linear(feat_dim, feat_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        level = F.gelu(self.fc1(z))
        level = self.fc2(level).view(-1, self.feat_dim, 1)
        return level.expand(-1, -1, self.seq_len)


# ============================================================
# Encoder
# Input: [B, 1, 2400]
# Output feature: [B, 512, 75]
# ============================================================
# depths: Tuple[int, int, int, int] = (2, 2, 4, 2)

class ECGEncoder1D(nn.Module):
    def __init__(self, cfg: ECGAEConfig):
        super().__init__()
        dims = cfg.dims
        depths = cfg.depths

        self.stem = nn.Sequential(
            nn.Conv1d(cfg.in_channels, dims[0], kernel_size=4, stride=4),   # 2400 -> 600
            LayerNorm1D(dims[0]),
        )

        self.stage1 = nn.Sequential(*[ConvNeXtV2Block1D(dims[0]) for _ in range(depths[0])])
        self.down1 = DownsampleBlock1D(dims[0], dims[1])   # 600 -> 300

        self.stage2 = nn.Sequential(*[ConvNeXtV2Block1D(dims[1]) for _ in range(depths[1])])
        self.down2 = DownsampleBlock1D(dims[1], dims[2])   # 300 -> 150

        self.stage3 = nn.Sequential(*[ConvNeXtV2Block1D(dims[2]) for _ in range(depths[2])])
        self.down3 = DownsampleBlock1D(dims[2], dims[3])   # 150 -> 75

        self.stage4 = nn.Sequential(*[ConvNeXtV2Block1D(dims[3]) for _ in range(depths[3])])
        self.bottleneck_attn = SelfAttention1D(
            dim=dims[3],
            num_heads=cfg.attn_heads,
            dropout=cfg.attn_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.down3(x)
        x = self.stage4(x)
        x = self.bottleneck_attn(x)
        return x   # [B, 512, 75]


class TemporalBottleneck(nn.Module):
    def __init__(self, in_dim: int, latent_channels: int):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv1d(in_dim, in_dim, kernel_size=3, padding=1),
            LayerNorm1D(in_dim),
            nn.GELU(),
        )
        self.to_latent = nn.Conv1d(in_dim, latent_channels, kernel_size=1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.pre(feat)
        z = self.to_latent(x)
        return z  # [B, latent_channels, 75]


# ============================================================
# Strong decoder
# z: [B, latent_channels, 75] -> ECG [B, 1, 2400]
# ============================================================
class ECGDecoder1D(nn.Module):
    def __init__(self, cfg: ECGAEConfig):
        super().__init__()
        dims = cfg.dims
        depths = cfg.depths

        self.in_proj = nn.Sequential(
            nn.Conv1d(cfg.latent_channels, dims[3], kernel_size=3, padding=1),
            LayerNorm1D(dims[3]),
            nn.GELU(),
        )

        self.decoder_bottleneck_attn = SelfAttention1D(
            dim=dims[3],
            num_heads=cfg.attn_heads,
            dropout=cfg.attn_dropout,
        )

        self.stage4 = nn.Sequential(
            *[ConvNeXtV2Block1D(dims[3]) for _ in range(depths[3])],
            ResidualConvBlock1D(dims[3], dims[3], kernel_size=15),
            ResidualConvBlock1D(dims[3], dims[3], kernel_size=15),
        )

        self.up3 = UpsampleBlock1D(dims[3], dims[2])  # 75 -> 150
        self.stage3 = nn.Sequential(
            *[ConvNeXtV2Block1D(dims[2]) for _ in range(depths[2])],
            ResidualConvBlock1D(dims[2], dims[2], kernel_size=15),
            ResidualConvBlock1D(dims[2], dims[2], kernel_size=15),
        )

        self.up2 = UpsampleBlock1D(dims[2], dims[1])  # 150 -> 300
        self.stage2 = nn.Sequential(
            *[ConvNeXtV2Block1D(dims[1]) for _ in range(depths[1])],
            ResidualConvBlock1D(dims[1], dims[1], kernel_size=15),
            ResidualConvBlock1D(dims[1], dims[1], kernel_size=15),
        )

        self.up1 = UpsampleBlock1D(dims[1], dims[0])  # 300 -> 600
        self.stage1 = nn.Sequential(
            *[ConvNeXtV2Block1D(dims[0]) for _ in range(depths[0])],
            ResidualConvBlock1D(dims[0], dims[0], kernel_size=15),
            ResidualConvBlock1D(dims[0], dims[0], kernel_size=15),
        )

        self.local_head = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="linear", align_corners=False),   # 600 -> 2400
            nn.Conv1d(dims[0], dims[0] // 2, kernel_size=7, padding=3),
            LayerNorm1D(dims[0] // 2),
            nn.GELU(),
            ResidualConvBlock1D(dims[0] // 2, dims[0] // 2, kernel_size=15),
            nn.Conv1d(dims[0] // 2, 1, kernel_size=7, padding=3),
        )

        # global refinement branch from pooled latent sequence
        pooled_dim = cfg.latent_channels * cfg.latent_length
        self.to_global = nn.Sequential(
            nn.Linear(pooled_dim, cfg.global_latent_dim),
            nn.GELU(),
            nn.Linear(cfg.global_latent_dim, cfg.global_latent_dim),
        )
        self.level_layer = LevelLayer(
            seq_len=cfg.input_length,
            feat_dim=1,
            latent_dim=cfg.global_latent_dim,
        )
        self.trend_layer = TrendLayer(
            seq_len=cfg.input_length,
            feat_dim=1,
            latent_dim=cfg.global_latent_dim,
            trend_poly=cfg.trend_poly,
        )

        self.out_refine = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=7, padding=3),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, latent_channels, 75]
        x = self.in_proj(z)
        x = self.decoder_bottleneck_attn(x)

        x = self.stage4(x)
        x = self.up3(x)
        x = self.stage3(x)
        x = self.up2(x)
        x = self.stage2(x)
        x = self.up1(x)
        x = self.stage1(x)

        local_out = self.local_head(x)   # [B, 1, 2400]

        z_flat = z.reshape(z.size(0), -1)
        z_global = self.to_global(z_flat)

        global_out = self.level_layer(z_global) + self.trend_layer(z_global)  # [B, 1, 2400]

        out = local_out + global_out
        out = self.out_refine(out)
        return out


# ============================================================
# ECG Standard Autoencoder (Không dùng VAE)
# ============================================================
class ECGAutoencoder(nn.Module):
    def __init__(self, cfg: ECGAEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = ECGEncoder1D(cfg)
        self.latent_head = TemporalBottleneck(cfg.dims[-1], cfg.latent_channels)
        self.decoder = ECGDecoder1D(cfg)

    def forward(self, ecg: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.encoder(ecg)                       # [B, 512, 75]
        z = self.latent_head(feat)                     # [B, latent_channels, 75]
        recon = self.decoder(z)

        return {
            "latent_ecg": z,               # [B, latent_channels, 75]
            "reconstructed_ecg": recon,    # [B, 1, 2400]
        }


# ============================================================
# Loss (Chỉ dùng L1 + Pearson, tối giản hóa)
# ============================================================
class ECGReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Khởi tạo duy nhất L1 Loss
        self.l1_loss = nn.L1Loss()

    @staticmethod
    def pearson_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        pred = pred.squeeze(1)
        target = target.squeeze(1)

        pred = pred - pred.mean(dim=1, keepdim=True)
        target = target - target.mean(dim=1, keepdim=True)

        num = (pred * target).sum(dim=1)
        den = torch.sqrt((pred.pow(2).sum(dim=1) + eps) * (target.pow(2).sum(dim=1) + eps))
        corr = num / den
        return 1.0 - corr.mean()

    def forward(self, outputs: Dict[str, torch.Tensor], target_ecg: torch.Tensor) -> Dict[str, torch.Tensor]:
        recon = outputs["reconstructed_ecg"]
        
        # 1. Tính L1 Loss
        loss_l1 = self.l1_loss(recon, target_ecg)
        
        # 2. Tính Pearson Loss
        loss_pearson = self.pearson_loss(recon, target_ecg)
        
        # 3. Tính Tổng Loss
        loss_total = loss_l1 + loss_pearson

        return {
            "loss_total": loss_total,
            "loss_l1": loss_l1,
            "loss_pearson": loss_pearson
        }


