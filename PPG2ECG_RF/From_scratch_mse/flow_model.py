import math
import torch
import torch.nn as nn

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

class FiLMBlock1D(nn.Module):
    def __init__(self, dim, time_emb_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim * 2)
        )
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, dim)
        self.norm2 = nn.GroupNorm(8, dim)
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        scale, shift = self.mlp(t_emb).unsqueeze(-1).chunk(2, dim=1)
        
        res = x
        x = self.norm1(x)
        x = x * (scale + 1) + shift  
        x = self.act(self.conv1(x))
        x = self.norm2(x)
        x = self.act(self.conv2(x))
        return x + res

class LatentRectifiedFlow(nn.Module):
    def __init__(self, latent_channels=16, cond_channels=16, hidden_dim=128, num_blocks=6):
        super().__init__()
        self.time_dim = hidden_dim * 4
        
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim)
        )

        in_channels = latent_channels + cond_channels
        self.conv_in = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1)
        
        self.blocks = nn.ModuleList([
            FiLMBlock1D(hidden_dim, self.time_dim) for _ in range(num_blocks)
        ])
        
        self.conv_out = nn.Conv1d(hidden_dim, latent_channels, kernel_size=3, padding=1)

    def forward(self, xt, t, z_ppg):
       
        t_emb = self.time_mlp(t)
        
        h = torch.cat([xt, z_ppg], dim=1)  # [Batch, 32, 75]
        h = self.conv_in(h)
        
        for block in self.blocks:
            h = block(h, t_emb)
            
        out_v = self.conv_out(h) 
        return out_v
    



