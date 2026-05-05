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

class ECG2ECGModel(nn.Module):
    def __init__(
        self, 
        in_channels: int = 1,
        base_dims: Tuple[int, int, int, int] = (64, 128, 256, 512),
        depths: Tuple[int, int, int, int] = (2, 2, 4, 2),
      
    ):
        super().__init__()
        self.decoder = ECGDecoder(base_dims)
        self.ecg_enc = ConvNeXtEncoder(in_channels, base_dims, depths)

    def forward(self, ecg: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        ecg_feat, skips = self.ecg_enc(ecg)
        recon_ecg = self.decoder(ecg_feat, skips)
        results = {"recon_ecg": recon_ecg}
            
        return results