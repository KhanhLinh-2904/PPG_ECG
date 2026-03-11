import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. THÊM CLASS NÀY ĐỂ FIX LỖI DIMENSION
class ChannelLayerNorm(nn.Module):
    """ Tự động xoay trục Channel xuống cuối để LayerNorm có thể tính toán """
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        # [Batch, Channel, Height, Width] -> [Batch, Height, Width, Channel]
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        # [Batch, Height, Width, Channel] -> [Batch, Channel, Height, Width]
        x = x.permute(0, 3, 1, 2)
        return x

class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) 
        self.act = nn.GELU()
        self.grn = GRN(4 * dim) 
        self.pwconv2 = nn.Linear(4 * dim, dim) 

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # Block này đã được xử lý xoay trục an toàn
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return input + x

class ConvNeXtV2_2D(nn.Module):
    def __init__(self, in_channels=2, output_dim=256, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        
        # 2. SỬA STEM: Thay nn.LayerNorm bằng ChannelLayerNorm
        stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            ChannelLayerNorm(dims[0]) 
        )
        self.downsample_layers.append(stem)
        
        # 3. SỬA DOWNSAMPLE: Thay nn.LayerNorm bằng ChannelLayerNorm
        for i in range(3):
            downsample_layer = nn.Sequential(
                ChannelLayerNorm(dims[i]), 
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtV2Block(dim=dims[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], output_dim)

    def forward(self, x):
        features = [] 
        
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            features.append(x) 
            
        latent = x.mean(dim=[2, 3]) 
        latent = self.norm(latent)
        latent = self.head(latent)
        
        return latent, features