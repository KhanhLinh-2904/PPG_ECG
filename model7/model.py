import copy
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from load_data import LoadData


# ============================================================
# 0. BASIC MODULES
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


class ConvNeXtV2Encoder(nn.Module):
    def __init__(self, in_channels: int = 32, dims: List[int] = [64, 128, 256]):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        self.stages = nn.ModuleList()

        self.downsample_layers.append(nn.Conv1d(in_channels, dims[0], kernel_size=4, stride=4))
        for i in range(len(dims)):
            self.stages.append(nn.Sequential(ConvNeXtV2Block1D(dims[i]), ConvNeXtV2Block1D(dims[i])))
            if i < len(dims) - 1:
                self.downsample_layers.append(nn.Sequential(LayerNorm1D(dims[i]), nn.Conv1d(dims[i], dims[i + 1], kernel_size=2, stride=2)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skips = []
        for i in range(len(self.stages)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            skips.append(x)
        return x, skips


class ResidualConvBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.need_proj = in_channels != out_channels
        self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=1) if self.need_proj else nn.Identity()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8 if out_channels >= 8 else 1, out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8 if out_channels >= 8 else 1, out_channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block(x) + self.proj(x))


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
        self.branch2 = nn.Sequential(nn.Conv1d(in_channels, branch_c, kernel_size=1), nn.Conv1d(branch_c, branch_c, kernel_size=3, padding=1))
        self.branch3 = nn.Sequential(nn.Conv1d(in_channels, branch_c, kernel_size=1), nn.Conv1d(branch_c, branch_c, kernel_size=5, padding=2))
        self.branch4 = nn.Sequential(nn.MaxPool1d(kernel_size=3, stride=1, padding=1), nn.Conv1d(in_channels, branch_c, kernel_size=1))
        self.out_norm = LayerNorm1D(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)
        return self.out_norm(x)


# ============================================================
# 1. BETTER FREQUENCY FUSION
# ============================================================
class FrequencyFusionV2(nn.Module):
    """
    Better than simple concat+1x1:
    - log FFT magnitude for stability
    - residual fusion
    - gating so frequency branch does not dominate
    """
    def __init__(self, dim: int):
        super().__init__()
        self.freq_proj = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1),
            LayerNorm1D(dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            LayerNorm1D(dim),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Conv1d(dim * 2, dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.out = ResidualConvBlock1D(dim, dim)

    def forward(self, time_feat: torch.Tensor) -> torch.Tensor:
        fft_complex = torch.fft.rfft(time_feat.float(), norm='ortho')
        fft_mag = torch.log1p(torch.abs(fft_complex))
        fft_mag = F.interpolate(fft_mag, size=time_feat.shape[-1], mode='linear', align_corners=False)
        freq_feat = self.freq_proj(fft_mag.to(time_feat.dtype))
        gate = self.gate(torch.cat([time_feat, freq_feat], dim=1))
        fused = time_feat + gate * freq_feat
        return self.out(fused)


# ============================================================
# 2. STRONGER DECODER
# ============================================================
class AttentionBlock(nn.Module):
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Conv1d(F_g, F_int, kernel_size=1)
        self.W_x = nn.Conv1d(F_l, F_int, kernel_size=1)
        self.psi = nn.Conv1d(F_int, 1, kernel_size=1)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = torch.sigmoid(self.psi(F.gelu(g1 + x1)))
        return x * psi


class UpBlock1D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
        self.att = AttentionBlock(F_g=out_channels, F_l=skip_channels, F_int=max(out_channels // 2, 8))
        self.conv = ResidualConvBlock1D(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-1], mode='linear', align_corners=False)
        skip = self.att(g=x, x=skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class AttentionUNetDecoderV2(nn.Module):
    def __init__(self, dims: List[int] = [64, 128, 256]):
        super().__init__()
        self.bottleneck_refine = nn.Sequential(
            ResidualConvBlock1D(dims[2], dims[2]),
            ResidualConvBlock1D(dims[2], dims[2]),
        )
        self.up2 = UpBlock1D(dims[2], dims[1], dims[1])
        self.up1 = UpBlock1D(dims[1], dims[0], dims[0])
        self.up_final = nn.Sequential(
            nn.ConvTranspose1d(dims[0], dims[0], kernel_size=4, stride=4),
            ResidualConvBlock1D(dims[0], dims[0]),
            nn.Conv1d(dims[0], 1, kernel_size=1),
        )

    def forward(self, deep_feat: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        x = self.bottleneck_refine(deep_feat)
        x = self.up2(x, skips[1])
        x = self.up1(x, skips[0])
        x = self.up_final(x)
        return x


# ============================================================
# 3. STAGE 1 ECG AUTOENCODER
# ============================================================
class Stage1_ECG_AutoEncoder(nn.Module):
    def __init__(self, dims: List[int] = [64, 128, 256]):
        super().__init__()
        self.ecg_stem = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.encoder = ConvNeXtV2Encoder(in_channels=32, dims=dims)
        self.decoder = AttentionUNetDecoderV2(dims=dims)

    def forward(self, ecg_real: torch.Tensor) -> torch.Tensor:
        x = self.ecg_stem(ecg_real)
        deep_feat, skips = self.encoder(x)
        return self.decoder(deep_feat, skips)


# ============================================================
# 4. IMPROVED STAGE 2 MODEL
# Key fixes:
# - reuse pretrained ECG decoder weights/structure
# - frozen ECG encoder as teacher
# - lightweight latent matching instead of contrastive shortcut
# - optional derivatives (can be disabled in config)
# ============================================================
class Stage2_PPG2ECG_Improved(nn.Module):
    def __init__(self, pretrained_ecg_model: Stage1_ECG_AutoEncoder, dims: List[int] = [64, 128, 256], use_derivatives: bool = False):
        super().__init__()
        self.use_derivatives = use_derivatives

        # frozen ECG teacher
        self.ecg_stem = copy.deepcopy(pretrained_ecg_model.ecg_stem)
        self.ecg_encoder = copy.deepcopy(pretrained_ecg_model.encoder)
        for p in self.ecg_stem.parameters():
            p.requires_grad = False
        for p in self.ecg_encoder.parameters():
            p.requires_grad = False

        # PPG branch
        self.prep = DerivativeInput() if use_derivatives else nn.Identity()
        ppg_in_channels = 3 if use_derivatives else 1
        self.ppg_stem = InceptionStem1D(in_channels=ppg_in_channels, out_channels=32) if use_derivatives else nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.ppg_encoder = ConvNeXtV2Encoder(in_channels=32, dims=dims)
        self.freq_fusion = FrequencyFusionV2(dim=dims[-1])

        # decoder initialized from pretrained ECG decoder structure/weights
        self.decoder = copy.deepcopy(pretrained_ecg_model.decoder)

        # bottleneck adapter to make PPG latent closer to ECG latent
        self.latent_adapter = nn.Sequential(
            ResidualConvBlock1D(dims[-1], dims[-1]),
            nn.Conv1d(dims[-1], dims[-1], kernel_size=1),
        )

    def encode_ecg_teacher(self, ecg_real: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            e = self.ecg_stem(ecg_real)
            deep_feat_ecg, _ = self.ecg_encoder(e)
        return deep_feat_ecg

    def forward(self, ppg: torch.Tensor, ecg_real: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        x = self.prep(ppg)
        x = self.ppg_stem(x)
        deep_feat_ppg, skips_ppg = self.ppg_encoder(x)
        fusion_feat = self.freq_fusion(deep_feat_ppg)
        # fusion_feat = self.latent_adapter(fusion_feat)
        ecg_hat = self.decoder(fusion_feat, skips_ppg)
        latent_ppg_student = self.encode_ecg_teacher(ecg_hat)
        
        out = {
            "ecg_hat": ecg_hat,
            "latent_ppg": latent_ppg_student,
        }
        if ecg_real is not None:
            out["latent_ecg_teacher"] = self.encode_ecg_teacher(ecg_real)
        return out


# ============================================================
# 5. LOSSES
# Key fixes:
# - use L1 + MSE + Pearson
# - latent matching to ECG teacher
# - optional feature consistency on reconstructed ECG
# ============================================================
class Stage2LossImproved(nn.Module):
    def __init__(
        self, 
        lambda_l1: float = 1.0, 
        lambda_mse: float = 0.5, 
        lambda_pearson: float = 1.0, 
        lambda_latent: float = 0.1,
        temperature: float = 0.1  # Thêm tham số nhiệt độ cho contrastive loss
    ):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_mse = lambda_mse
        self.lambda_pearson = lambda_pearson
        self.lambda_latent = lambda_latent
        self.temperature = temperature
        
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def pearson_correlation_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_mean = pred.mean(dim=-1, keepdim=True)
        target_mean = target.mean(dim=-1, keepdim=True)
        pred_centered = pred - pred_mean
        target_centered = target - target_mean
        cov = (pred_centered * target_centered).sum(dim=-1)
        var_pred = (pred_centered ** 2).sum(dim=-1)
        var_target = (target_centered ** 2).sum(dim=-1)
        pearson = cov / (torch.sqrt(var_pred * var_target) + 1e-8)
        return 1.0 - pearson.mean()

    def contrastive_latent_loss(self, latent_ppg: torch.Tensor, latent_ecg: torch.Tensor) -> torch.Tensor:
        """
        Sử dụng NT-Xent (Normalized Temperature-scaled Cross Entropy Loss)
        thay cho Cosine Similarity đơn thuần.
        """
        # 1. Global Average Pooling để đưa về vector 1D (Batch_Size, Channels)
        z_ppg = F.adaptive_avg_pool1d(latent_ppg, 1).squeeze(-1)
        z_ecg = F.adaptive_avg_pool1d(latent_ecg, 1).squeeze(-1)
        
        # 2. L2 Normalization (Bắt buộc cho contrastive loss)
        z_ppg = F.normalize(z_ppg, dim=-1)
        z_ecg = F.normalize(z_ecg, dim=-1)
        
        # 3. Tính ma trận độ tương đồng Cosine (Batch_Size x Batch_Size)
        # Chia cho temperature để làm sắc nét phân bố xác suất
        logits = torch.matmul(z_ppg, z_ecg.T) / self.temperature
        
        # 4. Tạo Labels
        # Cặp Positive là các phần tử trên đường chéo (z_ppg[i] match với z_ecg[i])
        labels = torch.arange(logits.size(0), device=logits.device)
        
        # 5. Dùng CrossEntropy (Ép đường chéo tiến tới 1, các ô khác tiến tới 0)
        loss = F.cross_entropy(logits, labels)
        
        return loss

    def forward(self, out: Dict[str, torch.Tensor], ecg_real: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        ecg_hat = out["ecg_hat"]
        
        # Tính các loss liên quan đến việc sinh lại tín hiệu
        loss_l1 = self.l1_loss(ecg_hat, ecg_real)
        loss_mse = self.mse_loss(ecg_hat, ecg_real)
        loss_pearson = self.pearson_correlation_loss(ecg_hat, ecg_real)

        total = self.lambda_l1 * loss_l1 + self.lambda_mse * loss_mse + self.lambda_pearson * loss_pearson
        metrics = {
            "l1": loss_l1,
            "mse": loss_mse,
            "pearson": loss_pearson,
        }

        # Nếu model trả về latent từ ECG Teacher (khi train)
        if "latent_ecg_teacher" in out:
            # SỬ DỤNG CONTRASTIVE LOSS TẠI ĐÂY
            loss_latent = self.contrastive_latent_loss(out["latent_ppg"], out["latent_ecg_teacher"])
            
            total = total + self.lambda_latent * loss_latent
            metrics["latent"] = loss_latent
        else:
            metrics["latent"] = torch.tensor(0.0, device=ecg_real.device)

        metrics["total"] = total
        return total, metrics


