import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

def ddpm_schedule(beta1: float, beta2: float, T: int) -> Dict[str, torch.Tensor]:
   
    assert beta1 < beta2 < 1.0

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    sqrtmab = torch.sqrt(1 - alphabar_t)

    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    return {
        "beta_t": beta_t,
        "alpha_t": alpha_t,  # \alpha_t
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
    }

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)
        
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose1d(
            in_channels, in_channels, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels*2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class SegmentUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose1d(
            in_channels, in_channels, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1):
        x1 = self.up(x1)
        x = self.conv(x1)
        return x
    
class ConditionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cuda"
        
        self.inc_c = DoubleConv(1, 64)
        self.inc_freq = DoubleConv(1, 64)

        self.down1_c = Down(64, 128)
        self.down2_c = Down(128, 256)
        self.down3_c = Down(256, 512)
        self.down4_c = Down(512, 1024)
        self.down5_c = Down(1024, 2048 // 2)
        
        self.up1_c = SegmentUp(1024, 512)
        self.up2_c = SegmentUp(512, 256)
        self.up3_c = SegmentUp(256, 128)
        self.up4_c = SegmentUp(128, 64)
        self.up5_c = SegmentUp(64, 32)

    def forward(self, x):
        d1 = self.inc_c(x)
        d2 = self.down1_c(d1)
        d3 = self.down2_c(d2)
        d4 = self.down3_c(d3)
        d5 = self.down4_c(d4)
        d6 = self.down5_c(d5)

        u1 = self.up1_c(d6)
        u2 = self.up2_c(u1)
        u3 = self.up3_c(u2)
        u4 = self.up4_c(u3)
        u5 = self.up5_c(u4)
      
        return {
            "down_conditions": [d1, d2, d3, d4, d5, d6],
            "up_conditions": [u1, u2, u3, u4, u5],
        }

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln = nn.LayerNorm([embed_dim], elementwise_affine=True)
        self.ff_cross = nn.Sequential(
            nn.LayerNorm([embed_dim], elementwise_affine=True),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x, c):
        x_ln = self.ln(x.permute(0, 2, 1))
        c_ln = self.ln(c.permute(0, 2, 1))
        attention_value, _ = self.cross_attention(x_ln, c_ln, c_ln)
        attention_value = attention_value + x_ln
        attention_value = self.ff_cross(attention_value) + attention_value
        return attention_value.permute(0, 2, 1)

class DiffusionUNetCrossAttention(nn.Module):
    def __init__(self, in_size, channels, device, num_heads=8):
        super().__init__()
        self.in_size = in_size
        self.channels = channels
        self.device = device
        
        self.inc_x = DoubleConv(channels, 64)
        self.inc_freq = DoubleConv(channels, 64)

        self.down1_x = Down(64, 128)
        self.down2_x = Down(128, 256)
        self.down3_x = Down(256, 512)
        self.down4_x = Down(512, 1024)
        self.down5_x = Down(1024, 2048 // 2)
        
        self.up1_x = Up(1024, 512)
        self.up2_x = Up(512, 256)
        self.up3_x = Up(256, 128)
        self.up4_x = Up(128, 64)
        self.up5_x = Up(64, 32)
        
        self.cross_attention_down1 = CrossAttentionBlock(64, num_heads)
        self.cross_attention_down2 = CrossAttentionBlock(128, num_heads)
        self.cross_attention_down3 = CrossAttentionBlock(256, num_heads)
        self.cross_attention_down4 = CrossAttentionBlock(512, num_heads)
        self.cross_attention_down5 = CrossAttentionBlock(1024, num_heads)
        self.cross_attention_down6 = CrossAttentionBlock(1024, num_heads)

        self.cross_attention_up1 = CrossAttentionBlock(512, num_heads)
        self.cross_attention_up2 = CrossAttentionBlock(256, num_heads)
        self.cross_attention_up3 = CrossAttentionBlock(128, num_heads)
        self.cross_attention_up4 = CrossAttentionBlock(64, num_heads)
        self.cross_attention_up5 = CrossAttentionBlock(32, num_heads)

        self.outc_x = nn.Conv1d(32, channels, kernel_size=1)

    def pos_encoding(self, t, channels, embed_size):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc.view(-1, channels, 1).repeat(1, 1, embed_size)
    
    def forward(self, x, c, t):
       
        t = t.unsqueeze(-1)
        x1 = self.inc_x(x)
        x1 = self.cross_attention_down1(x1, c["down_conditions"][0])
        
        x2 = self.down1_x(x1) + self.pos_encoding(t, 128, x1.shape[-1] // 2)
        x2 = self.cross_attention_down2(x2, c["down_conditions"][1])

        x3 = self.down2_x(x2) + self.pos_encoding(t, 256, x1.shape[-1] // 4)
        x3 = self.cross_attention_down3(x3, c["down_conditions"][2])
        
        x4 = self.down3_x(x3) + self.pos_encoding(t, 512, x1.shape[-1] // 8)
        x4 = self.cross_attention_down4(x4, c["down_conditions"][3])

        x5 = self.down4_x(x4) + self.pos_encoding(t, 1024, x1.shape[-1] // 16)
        x5 = self.cross_attention_down5(x5, c["down_conditions"][4])

        x6 = self.down5_x(x5) + self.pos_encoding(t, 1024, x1.shape[-1] // 32)
        x6 = self.cross_attention_down6(x6, c["down_conditions"][5])
        
        x = self.up1_x(x6, x5) + self.pos_encoding(t, 512, x1.shape[-1] // 16)
        x = self.cross_attention_up1(x, c["up_conditions"][0])

        x = self.up2_x(x, x4) + self.pos_encoding(t, 256, x1.shape[-1] // 8)
        x = self.cross_attention_up2(x, c["up_conditions"][1])

        x = self.up3_x(x, x3) + self.pos_encoding(t, 128, x1.shape[-1] // 4)
        x = self.cross_attention_up3(x, c["up_conditions"][2])

        x = self.up4_x(x, x2) + self.pos_encoding(t, 64, x1.shape[-1] // 2)
        x = self.cross_attention_up4(x, c["up_conditions"][3])

        x = self.up5_x(x, x1) + self.pos_encoding(t, 32, x1.shape[-1])
        x = self.cross_attention_up5(x, c["up_conditions"][4])

        output = self.outc_x(x)

        return output.view(-1, self.channels, output.shape[-1])
    
