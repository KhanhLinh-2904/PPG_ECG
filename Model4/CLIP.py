import math
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm1D(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class GRN1D(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=2, keepdim=True)          # [B, C, 1]
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)   # [B, C, 1]
        return self.gamma * (x * nx) + self.beta + x


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x / keep_prob * random_tensor


class ConvNeXtBlock1D(nn.Module):
    def __init__(self, dim: int, expansion: int = 4, drop_path: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm1D(dim)
        self.pwconv1 = nn.Conv1d(dim, dim * expansion, kernel_size=1)
        self.act = nn.GELU()
        self.grn = GRN1D(dim * expansion)
        self.pwconv2 = nn.Conv1d(dim * expansion, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = residual + self.drop_path(x)
        return x


class MultiKernelStem1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        c1 = out_ch // 3
        c2 = out_ch // 3
        c3 = out_ch - c1 - c2

        self.b1 = nn.Sequential(
            nn.Conv1d(in_ch, c1, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(c1),
            nn.GELU(),
        )
        self.b2 = nn.Sequential(
            nn.Conv1d(in_ch, c2, kernel_size=7, stride=stride, padding=3),
            nn.BatchNorm1d(c2),
            nn.GELU(),
        )
        self.b3 = nn.Sequential(
            nn.Conv1d(in_ch, c3, kernel_size=15, stride=stride, padding=7),
            nn.BatchNorm1d(c3),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=1),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
        return self.fuse(x)


class DownsampleBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_blocks: int = 2, drop_path: float = 0.0):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[
            ConvNeXtBlock1D(out_ch, drop_path=drop_path) for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        x = self.blocks(x)
        return x


class UpsampleBlock1D(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, num_blocks: int = 2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[
            ConvNeXtBlock1D(out_ch) for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.proj(x)
        x = self.blocks(x)
        return x


class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class SelfAttentionBlock1D(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.pos = PositionalEncoding1D(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt = x.transpose(1, 2)
        xt = self.pos(xt)

        h = self.norm1(xt)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        xt = xt + attn_out

        xt = xt + self.mlp(self.norm2(xt))
        return xt.transpose(1, 2)



def first_derivative(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([torch.zeros_like(x[:, :, :1]), x[:, :, 1:] - x[:, :, :-1]], dim=2)


def second_derivative(x: torch.Tensor) -> torch.Tensor:
    dx = first_derivative(x)
    return first_derivative(dx)


def moving_average(x: torch.Tensor, kernel_size: int = 15) -> torch.Tensor:
    pad = kernel_size // 2
    weight = torch.ones(1, 1, kernel_size, device=x.device, dtype=x.dtype) / kernel_size
    return F.conv1d(F.pad(x, (pad, pad), mode="reflect"), weight)


def z_norm_channelwise(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = x.mean(dim=2, keepdim=True)
    std = x.std(dim=2, keepdim=True) + eps
    return (x - mean) / std


class SharedPPGStem(nn.Module):
    def __init__(
        self,
        in_ch: int = 5,
        out_ch: int = 48,
        num_blocks: int = 2,
        drop_path: float = 0.05,
    ):
        super().__init__()
        self.stem = MultiKernelStem1D(in_ch, out_ch, stride=2)
        self.blocks = nn.Sequential(*[
            ConvNeXtBlock1D(out_ch, drop_path=drop_path) for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return x

class ECGComponentEncoderBranch(nn.Module):

    def __init__(
        self,
        in_ch: int,
        dims: Tuple[int, int, int] = (96, 128, 256),
        num_blocks_per_stage: int = 2,
        num_heads: int = 8,
        attn_depth: int = 2,
        drop_path: float = 0.05,
    ):
        super().__init__()
        d2, d3, d4 = dims

        self.adapter = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, kernel_size=1),
            nn.BatchNorm1d(in_ch),
            nn.GELU(),
        )

        self.down2 = DownsampleBlock1D(in_ch, d2, num_blocks=num_blocks_per_stage, drop_path=drop_path)
        self.down3 = DownsampleBlock1D(d2, d3, num_blocks=num_blocks_per_stage, drop_path=drop_path)
        self.down4 = DownsampleBlock1D(d3, d4, num_blocks=num_blocks_per_stage, drop_path=drop_path)

        self.bottleneck = nn.Sequential(*[
            SelfAttentionBlock1D(d4, num_heads=num_heads) for _ in range(attn_depth)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        s1 = self.adapter(x)   # 1/2
        s2 = self.down2(s1)    # 1/4
        s3 = self.down3(s2)    # 1/8
        s4 = self.down4(s3)    # 1/16
        z = self.bottleneck(s4)

        skips = [s1, s2, s3, s4]
        return z, skips


class ECGComponentDecoder(nn.Module):
   
    def __init__(self, dims: Tuple[int, int, int, int] = (48, 96, 128, 256), out_ch: int = 1):
        super().__init__()
        d1, d2, d3, d4 = dims

        self.up3 = UpsampleBlock1D(d4, d4, d3)
        self.up2 = UpsampleBlock1D(d3, d3, d2)
        self.up1 = UpsampleBlock1D(d2, d2, d1)

        self.up0 = nn.Sequential(
            nn.Conv1d(d1, d1, kernel_size=3, padding=1),
            nn.BatchNorm1d(d1),
            nn.GELU(),
            ConvNeXtBlock1D(d1),
        )

        self.final = nn.Sequential(
            nn.Conv1d(d1, d1 // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d1 // 2, out_ch, kernel_size=7, padding=3),
        )

    def forward(self, z: torch.Tensor, skips: List[torch.Tensor], out_len: int) -> torch.Tensor:
        s1, s2, s3, s4 = skips

        x = self.up3(z, s4)
        x = self.up2(x, s3)
        x = self.up1(x, s2)

        x = F.interpolate(x, size=s1.shape[-1], mode="linear", align_corners=False)
        x = x + s1
        x = self.up0(x)

        x = F.interpolate(x, size=out_len, mode="linear", align_corners=False)
        y = self.final(x)
        return y



class ECGFusionHead(nn.Module):

    def __init__(self, hidden_ch: int = 32):
        super().__init__()

        # [qrs, non, qrs+non, qrs-non] => 4 channels
        self.fusion = nn.Sequential(
            nn.Conv1d(4, hidden_ch, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_ch),
            nn.GELU(),
            nn.Conv1d(hidden_ch, hidden_ch, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_ch),
            nn.GELU(),
        )

        self.gate_head = nn.Sequential(
            nn.Conv1d(hidden_ch, hidden_ch // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_ch // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.residual_head = nn.Sequential(
            nn.Conv1d(hidden_ch, hidden_ch // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_ch // 2, 1, kernel_size=1),
        )

    def forward(self, ecg_qrs_pred: torch.Tensor, ecg_non_qrs_pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = torch.cat([
            ecg_qrs_pred,
            ecg_non_qrs_pred,
            ecg_qrs_pred + ecg_non_qrs_pred,
            ecg_qrs_pred - ecg_non_qrs_pred,
        ], dim=1)

        h = self.fusion(feat)
        gate = self.gate_head(h)              # [B,1,L]
        residual = self.residual_head(h)      # [B,1,L]

        ecg_final = gate * ecg_qrs_pred + (1.0 - gate) * ecg_non_qrs_pred + residual

        return {
            "ecg_final_pred": ecg_final,
            "gate": gate,
            "residual": residual,
        }



class PPGtoECGDualBranchReconstructionNet(nn.Module):
  
    def __init__(
        self,
        input_len: int = 2400,
        dims: Tuple[int, int, int, int] = (48, 96, 128, 256),
        num_blocks_per_stage: int = 2,
        num_heads: int = 8,
        attn_depth_qrs: int = 2,
        attn_depth_non_qrs: int = 2,
        drop_path: float = 0.05,
    ):
        super().__init__()
        self.input_len = input_len
        d1, d2, d3, d4 = dims

        # Shared input: [ppg_raw, vpg, apg, smooth, trend]
        self.shared_stem = SharedPPGStem(
            in_ch=5,
            out_ch=d1,
            num_blocks=num_blocks_per_stage,
            drop_path=drop_path,
        )

        # QRS branch
        self.qrs_encoder = ECGComponentEncoderBranch(
            in_ch=d1,
            dims=(d2, d3, d4),
            num_blocks_per_stage=num_blocks_per_stage,
            num_heads=num_heads,
            attn_depth=attn_depth_qrs,
            drop_path=drop_path,
        )
        self.qrs_decoder = ECGComponentDecoder(
            dims=dims,
            out_ch=1,
        )

        # non-QRS branch
        self.non_qrs_encoder = ECGComponentEncoderBranch(
            in_ch=d1,
            dims=(d2, d3, d4),
            num_blocks_per_stage=num_blocks_per_stage,
            num_heads=num_heads,
            attn_depth=attn_depth_non_qrs,
            drop_path=drop_path,
        )
        self.non_qrs_decoder = ECGComponentDecoder(
            dims=dims,
            out_ch=1,
        )

        # final fusion
        self.fusion_head = ECGFusionHead(hidden_ch=32)

    def build_shared_input(self, ppg_raw: torch.Tensor) -> torch.Tensor:
        x = ppg_raw
        vpg = z_norm_channelwise(first_derivative(x))
        apg = z_norm_channelwise(second_derivative(x))
        smooth = moving_average(x, kernel_size=11)
        trend = moving_average(x, kernel_size=31)

        shared_input = torch.cat([x, vpg, apg, smooth, trend], dim=1)
        return shared_input

    def encode(self, ppg_raw: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared_input = self.build_shared_input(ppg_raw)
        shared_feat = self.shared_stem(shared_input)

        z_qrs, qrs_skips = self.qrs_encoder(shared_feat)
        z_non_qrs, non_qrs_skips = self.non_qrs_encoder(shared_feat)

        return {
            "shared_feat": shared_feat,
            "z_qrs": z_qrs,
            "z_non_qrs": z_non_qrs,
            "qrs_skips": qrs_skips,
            "non_qrs_skips": non_qrs_skips,
        }

    def decode(self, encoded: Dict[str, torch.Tensor], out_len: int) -> Dict[str, torch.Tensor]:
        ecg_qrs_pred = self.qrs_decoder(
            encoded["z_qrs"],
            encoded["qrs_skips"],
            out_len=out_len,
        )

        ecg_non_qrs_pred = self.non_qrs_decoder(
            encoded["z_non_qrs"],
            encoded["non_qrs_skips"],
            out_len=out_len,
        )

        fused = self.fusion_head(ecg_qrs_pred, ecg_non_qrs_pred)

        return {
            "ecg_qrs_pred": ecg_qrs_pred,
            "ecg_non_qrs_pred": ecg_non_qrs_pred,
            **fused,
        }

    def forward(self, ppg_raw: torch.Tensor, return_features: bool = True) -> Dict[str, torch.Tensor]:
        assert ppg_raw.ndim == 3 and ppg_raw.shape[1] == 1, \
            f"ppg_raw phải có shape [B,1,L], got {ppg_raw.shape}"

        encoded = self.encode(ppg_raw)
        decoded = self.decode(encoded, out_len=ppg_raw.shape[-1])

        out = {**decoded}
        if return_features:
            out.update(encoded)
        return out

