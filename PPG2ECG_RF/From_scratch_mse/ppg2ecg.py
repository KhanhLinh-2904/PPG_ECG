import copy
import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class DownsampleBlock1D(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.block = nn.Sequential(
            LayerNorm1D(in_dim),
            nn.Conv1d(in_dim, out_dim, kernel_size=2, stride=2),
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
        x_seq = x.transpose(1, 2)  # [B, L, C]

        x_norm = self.norm1(x_seq)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_seq = x_seq + attn_out

        x_ffn = self.ffn(self.norm2(x_seq))
        x_seq = x_seq + x_ffn

        return x_seq.transpose(1, 2)  # [B, C, L]


# ============================================================
# PPG front-end
# ============================================================
class DerivativeInput(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = torch.gradient(x, dim=-1)[0]
        d2 = torch.gradient(d1, dim=-1)[0]
        return torch.cat([x, d1, d2], dim=1)


class InceptionStem1D(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 32):
        super().__init__()
        branch_c = out_channels // 4

        self.branch1 = nn.Conv1d(in_channels, branch_c, kernel_size=1)

        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, branch_c, kernel_size=1),
            nn.Conv1d(branch_c, branch_c, kernel_size=3, padding=1),
        )

        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, branch_c, kernel_size=1),
            nn.Conv1d(branch_c, branch_c, kernel_size=5, padding=2),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, branch_c, kernel_size=1),
        )

        self.out_norm = LayerNorm1D(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([
            self.branch1(x), self.branch2(x), 
            self.branch3(x), self.branch4(x)
        ], dim=1)
        return self.out_norm(x)


# ============================================================
# Config
# ============================================================
@dataclass
class PPG2ECGConfig:
    input_length: int = 2400
    ppg_in_channels: int = 1

    dims: Tuple[int, int, int, int] = (64, 128, 256, 512)
    depths: Tuple[int, int, int, int] = (2, 2, 4, 2)

    latent_channels: int = 16
    latent_length: int = 75

    attn_heads: int = 8
    attn_dropout: float = 0.0
    use_derivatives: bool = True


# ============================================================
# PPG Encoder & Heads
# ============================================================
class PPGEncoder1D(nn.Module):
    def __init__(self, cfg: PPG2ECGConfig):
        super().__init__()
        dims = cfg.dims
        depths = cfg.depths

        self.prep = DerivativeInput() if cfg.use_derivatives else nn.Identity()
        stem_in = 3 if cfg.use_derivatives else cfg.ppg_in_channels

        self.stem = InceptionStem1D(in_channels=stem_in, out_channels=32)

        self.patchify = nn.Sequential(
            nn.Conv1d(32, dims[0], kernel_size=4, stride=4),  # 2400 -> 600
            LayerNorm1D(dims[0]),
        )

        self.stage1 = nn.Sequential(*[ConvNeXtV2Block1D(dims[0]) for _ in range(depths[0])])
        self.down1 = DownsampleBlock1D(dims[0], dims[1])

        self.stage2 = nn.Sequential(*[ConvNeXtV2Block1D(dims[1]) for _ in range(depths[1])])
        self.down2 = DownsampleBlock1D(dims[1], dims[2])

        self.stage3 = nn.Sequential(*[ConvNeXtV2Block1D(dims[2]) for _ in range(depths[2])])
        self.down3 = DownsampleBlock1D(dims[2], dims[3])

        self.stage4 = nn.Sequential(*[ConvNeXtV2Block1D(dims[3]) for _ in range(depths[3])])
        self.bottleneck_attn = SelfAttention1D(
            dim=dims[3],
            num_heads=cfg.attn_heads,
            dropout=cfg.attn_dropout,
        )

    def forward(self, ppg: torch.Tensor) -> torch.Tensor:
        x = self.prep(ppg)
        x = self.stem(x)
        x = self.patchify(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.down3(x)
        x = self.stage4(x)
        x = self.bottleneck_attn(x)
        return x  # [B, 512, 75]


class TemporalBottleneck(nn.Module):
    def __init__(self, in_dim: int, latent_channels: int):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv1d(in_dim, in_dim, kernel_size=3, padding=1),
            LayerNorm1D(in_dim),
            nn.GELU(),
        )
        self.to_latent = nn.Conv1d(in_dim, latent_channels, kernel_size=1)

    def forward(self, feat: torch.Tensor):
        x = self.pre(feat)
        z = self.to_latent(x)
        return z


# ============================================================
# Main model (Alignment Phase Only)
# ============================================================
class PPG2ECGModel(nn.Module):
    def __init__(self, ecg_ae: nn.Module, cfg: PPG2ECGConfig):
        super().__init__()
        self.cfg = cfg

        # 1. Initialize Teacher (Pre-trained ECG Autoencoder)
        self.ecg_encoder = copy.deepcopy(ecg_ae.encoder)
        self.ecg_latent_head = copy.deepcopy(ecg_ae.latent_head)
        self.ecg_decoder = copy.deepcopy(ecg_ae.decoder) # Keep for inference if needed

        # Completely freeze Teacher
        for p in self.ecg_encoder.parameters():
            p.requires_grad = False
        for p in self.ecg_latent_head.parameters():
            p.requires_grad = False
        for p in self.ecg_decoder.parameters():
            p.requires_grad = False

        # 2. Initialize Student (PPG Branch)
        self.ppg_encoder = PPGEncoder1D(cfg)
        self.ppg_latent_head = TemporalBottleneck(cfg.dims[-1], cfg.latent_channels)

    @staticmethod
    def pool_temporal(x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool1d(x, 1).squeeze(-1)

    @torch.no_grad()
    def encode_ecg_teacher(self, ecg: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat_ecg = self.ecg_encoder(ecg)
        z_ecg = self.ecg_latent_head(feat_ecg)

        return {
            "z_ecg": z_ecg,
        }

    def forward(self, ppg: torch.Tensor, ecg: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # Process PPG branch
        feat_ppg = self.ppg_encoder(ppg)
        z_ppg = self.ppg_latent_head(feat_ppg)

        outputs = {
            "z_ppg": z_ppg,
        }

        # If ECG is provided (During Training), process ECG Teacher branch
        if ecg is not None:
            outputs.update(self.encode_ecg_teacher(ecg))

        return outputs


class CardioAlignLoss(nn.Module):
    """
    Loss function to align Latent space between PPG and ECG:
    1. CORAL Loss: Synchronize covariance matrix structure.
    2. KL Divergence: Force batch statistical distributions to be similar.
     
    """
    def __init__(
        self,
        coral_weight: float = 5.0,
        kl_weight: float = 0.1,  # Đã giảm từ 1.0 xuống 0.1 theo khuyến nghị
        eps: float = 1e-6,
    ):
        super().__init__()
        self.coral_weight = coral_weight
        self.kl_weight = kl_weight
        self.eps = eps
        
    def coral_loss(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """ Calculate CORAL across all data dimensions of the batch [B, C*L] """
        source = source.flatten(1)
        target = target.flatten(1)

        source = source - source.mean(dim=0, keepdim=True)
        target = target - target.mean(dim=0, keepdim=True)

        bs = source.size(0)
        bt = target.size(0)

        cov_source = (source.T @ source) / max(bs - 1, 1)
        cov_target = (target.T @ target) / max(bt - 1, 1)

        d = source.size(1)
        return torch.sum((cov_source - cov_target) ** 2) / (4.0 * d * d + self.eps)


    def kl_divergence_batch(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """ Calculate KL Divergence based on Batch Mean and Batch Variance statistics """
        source = source.flatten(1)
        target = target.flatten(1)
        
    
        mean_p = source.mean(dim=0)
        var_p = source.var(dim=0, unbiased=False) + self.eps
        
        mean_e = target.mean(dim=0).detach()
        var_e = target.var(dim=0, unbiased=False).detach() + self.eps
        
        kl = 0.5 * (torch.log(var_e / var_p) + (var_p + (mean_p - mean_e)**2) / var_e - 1.0)
        return torch.mean(kl)

    def forward(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        z_ppg = outputs["z_ppg"]
        z_ecg = outputs["z_ecg"].detach()

        # 1. Calculate CORAL Loss
        loss_coral = self.coral_loss(z_ppg, z_ecg)
        
        # 2. Calculate KL Divergence
        loss_kl = self.kl_divergence_batch(z_ppg, z_ecg)
        
        # Aggregate loss based on weights
        loss_total = (
            self.coral_weight * loss_coral
            + self.kl_weight * loss_kl
        )

        return {
            "loss_total": loss_total,
            "loss_coral": loss_coral,
            "loss_kl": loss_kl,
        }

