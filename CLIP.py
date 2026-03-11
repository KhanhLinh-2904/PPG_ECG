import torch
import torch.nn as nn
import torch.nn.functional as F
from ConvNeXt import ConvNeXtV2_2D
from Mamba import MambaDecoder_1D
import numpy as np
from stockwell import st

def stockwell_transform_v2(signal, fs, fmin, fmax):
    signal_length = len(signal)
    df = fs / signal_length
    
    fmin_samples = int(np.floor(fmin / df))
    fmax_samples = int(np.ceil(fmax / df))
    
    max_nyquist_sample = int(np.floor((fs / 2) / df))
    fmax_samples = min(fmax_samples, max_nyquist_sample)
    
    trans_signal = st.st(signal, fmin_samples, fmax_samples)
    return trans_signal

class PPG2ECG_Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.teacher_encoder = ConvNeXtV2_2D(in_channels=2, output_dim=256)
        self.student_encoder = ConvNeXtV2_2D(in_channels=2, output_dim=256)
        
        self.shared_projection = nn.Sequential(
            nn.Linear(256, 256), 
            nn.GELU(), 
            nn.Linear(256, 128)
        )
        
        self.mamba_decoder = MambaDecoder_1D(input_dim=128, output_length=2400, d_model=128)

    def forward(self, ppg_st_2d, ecg_st_2d=None):
        if ecg_st_2d is not None:
            with torch.no_grad():
                ecg_latent, _ = self.teacher_encoder(ecg_st_2d)
                ecg_embed = self.shared_projection(ecg_latent)
        else:
            ecg_embed = None
            
        ppg_latent, ppg_skips = self.student_encoder(ppg_st_2d)
        ppg_embed = self.shared_projection(ppg_latent)
        
        ppg_embed_for_decoder = ppg_embed.detach()
        ppg_skips_detached = [skip.detach() for skip in ppg_skips]
    
        reconstructed_ecg_1d = self.mamba_decoder(ppg_embed_for_decoder, ppg_skips_detached)
        
        if ecg_embed is not None:
            return ppg_embed, ecg_embed, reconstructed_ecg_1d
        
        return reconstructed_ecg_1d