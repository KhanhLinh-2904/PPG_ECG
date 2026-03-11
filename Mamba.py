import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaBlock_1D(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner, 
            kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1
        )

        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        A = torch.arange(1, self.d_state + 1).float().repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        b, l, d = x.shape

        x_and_res = self.in_proj(x)
        x_main, res = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        x_main = x_main.transpose(1, 2)
        x_main = self.conv1d(x_main)[:, :, :l] 
        x_main = x_main.transpose(1, 2)
        x_main = F.silu(x_main)

        A = -torch.exp(self.A_log.float()) 
        D = self.D.float()                 

        x_dbl = self.x_proj(x_main)
        delta, B, C = x_dbl.split([1, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))

        y = self.selective_scan(x_main, delta, A, B, C, D)

        y = y * F.silu(res)
        out = self.out_proj(y)
        return out

    def selective_scan(self, u, delta, A, B, C, D):
        b, l, d_in = u.shape
        d_state = A.shape[1]

        h = torch.zeros((b, d_in, d_state), device=u.device)
        ys = []

        for i in range(l):
            delta_i = delta[:, i, :].unsqueeze(-1)
            u_i = u[:, i, :].unsqueeze(-1)
            B_i = B[:, i, :].unsqueeze(1)
            C_i = C[:, i, :].unsqueeze(1)

            deltaA_i = torch.exp(delta_i * A)
            deltaB_i = delta_i * B_i * u_i

            h = deltaA_i * h + deltaB_i
            
            y_i = (h * C_i).sum(dim=-1) + u[:, i, :] * D
            ys.append(y_i)

        return torch.stack(ys, dim=1)
    
class FrequencyPoolingBridge(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.channel_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, feature_map_2d, target_time_steps):
       
        # 1.  Frequency (dim=2)  Average Pooling
        # [B, C, F, T] -> [B, C, T]
        pooled_1d = torch.mean(feature_map_2d, dim=2) 
        
        # 2.  (Channels)
        # [B, C, T] -> [B, out_channels, T]
        proj_1d = self.channel_proj(pooled_1d)
        
        # 3. (Time) align with Mamba 
        if proj_1d.shape[-1] != target_time_steps:
            proj_1d = F.interpolate(proj_1d, size=target_time_steps, mode='linear', align_corners=False)
            
        # 4. Mamba: [Batch, Time, Channels]
        mamba_input = proj_1d.permute(0, 2, 1)
        
        return mamba_input

class MambaDecoder_1D(nn.Module):
    def __init__(self, input_dim=128, output_length=2400, d_model=128, encoder_dims=[96, 192, 384, 768]):
        super().__init__()
        self.d_model = d_model
        self.seed_length = 150 
        self.latent_expander = nn.Linear(input_dim, self.seed_length * d_model)
        
        #  Stage from ConvNeXt into Mamba)
        # ConvNeXt dims : Stage 0(96), Stage 1(192), Stage 2(384), Stage 3(768)
        self.bridge_skip3 = FrequencyPoolingBridge(encoder_dims[3], d_model) 
        self.bridge_skip2 = FrequencyPoolingBridge(encoder_dims[2], d_model) 
        self.bridge_skip1 = FrequencyPoolingBridge(encoder_dims[1], d_model) 
        
        self.mamba1 = MambaBlock_1D(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.mamba2 = MambaBlock_1D(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.mamba3 = MambaBlock_1D(d_model=d_model, d_state=16, d_conv=4, expand=2)
        
        self.final_head = nn.Linear(d_model, 1)

    def forward(self, latent_vec, skip_features):
        seq = self.latent_expander(latent_vec)
        seq = seq.view(seq.size(0), self.seed_length, self.d_model) # [Batch, 150, 128]
        
        #  Skip Features from ConvNeXt (0 is shallow, 3 is deepest)
        feat0, feat1, feat2, feat3 = skip_features
        
        # Add Skip Connection 
        seq = seq + self.bridge_skip3(feat3, target_time_steps=seq.shape[1])
        seq = self.mamba1(seq)
        
        # Upsample 4x (150 -> 600)
        seq = seq.transpose(1, 2)
        seq = F.interpolate(seq, scale_factor=4, mode='linear', align_corners=False)
        seq = seq.transpose(1, 2)
        
        # ---  2 (Từ 600 step) ---
        seq = seq + self.bridge_skip2(feat2, target_time_steps=seq.shape[1])
        seq = self.mamba2(seq)
        
        # Upsample 4x (600 -> 2400)
        seq = seq.transpose(1, 2)
        seq = F.interpolate(seq, scale_factor=4, mode='linear', align_corners=False)
        seq = seq.transpose(1, 2)
        
        # ---  3 (2400 step) ---
        seq = seq + self.bridge_skip1(feat1, target_time_steps=seq.shape[1])
        seq = self.mamba3(seq)
        
        #  ECG 1D
        ecg_out = self.final_head(seq).squeeze(-1) 
        return ecg_out