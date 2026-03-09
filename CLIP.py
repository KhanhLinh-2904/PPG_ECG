import torch
from torch import nn
import math
from ResNet50 import ResNet50_1D

OUTPUT_EMBED_DIM = 128 
INPUT_LENGTH = 2400
BASE_WIDTH = 64
EXPANSION = 4 
LAYERS = [3, 4, 6, 3] 
class ECGEssembleCLIP(nn.Module):
    def __init__(self, embed_dim=OUTPUT_EMBED_DIM):
        super(ECGEssembleCLIP, self).__init__()

        self.encode_ecg = ResNet50_1D(
        layers=[3, 4, 6, 3] ,
        num_classes=OUTPUT_EMBED_DIM)


        self.encode_ppg = ResNet50_1D(
        layers=[3, 4, 6, 3] ,
        num_classes=OUTPUT_EMBED_DIM)

        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def forward(self, ecg_original, ppg_original):
        if ecg_original is None:
            ecg_predicted_features, featured_PPG, feature_lists_PPG = self.encode_ppg(ppg_original)
            return  featured_PPG, feature_lists_PPG
        else:
            ecg_original_features, featured_ECG, feature_lists_ECG = self.encode_ecg(ecg_original)
            ecg_predicted_features, featured_PPG, feature_lists_PPG = self.encode_ppg(ppg_original)
            ecg_original_features = ecg_original_features / ecg_original_features.norm(dim=1, keepdim=True)
            ecg_predicted_features = ecg_predicted_features / ecg_predicted_features.norm(dim=1, keepdim=True)

            logit_scale = self.logit_scale.exp()
        
            logits_per_original = logit_scale * ecg_original_features @ ecg_predicted_features.t()
            
            return logits_per_original, featured_PPG, feature_lists_PPG


class DecoderBlock_UNet(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='linear', align_corners=False)
        
        total_in_channels = in_channels + skip_channels
        
        self.conv = nn.Sequential(
            nn.Conv1d(total_in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip=None):
        x = self.upsample(x)
        
        if skip is not None:
            if x.size(2) != skip.size(2):
                x = nn.functional.interpolate(x, size=skip.size(2), mode='nearest')
            
            #  channel: [Batch, C_x, L] + [Batch, C_skip, L] -> [Batch, C_x + C_skip, L]
            x = torch.cat([x, skip], dim=1) 
            
        return self.conv(x)
class ECGDecoder_UNet(nn.Module):
    def __init__(self, bottleneck_channels=2048,  target_length=2400):
        super().__init__()
        print("target_length: ", target_length)
        self.target_length = target_length
        # Adapter: Bottleneck 
        self.adapter = nn.Sequential(
            nn.Conv1d(bottleneck_channels, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        ) 

        
        # Block 1: Input 512 | Skip f3 (1024) -> Out 256
        self.block1 = DecoderBlock_UNet(in_channels=512, out_channels=256, skip_channels=1024)
        
        # Block 2: Input 256 | Skip f2 (512) -> Out 128
        self.block2 = DecoderBlock_UNet(in_channels=256, out_channels=128, skip_channels=512)
        
        # Block 3: Input 128 | Skip f1 (256) -> Out 64
        self.block3 = DecoderBlock_UNet(in_channels=128, out_channels=64, skip_channels=256)
        
        # Các block cuối để upsample về độ dài gốc (không còn skip connection)
        self.block4 = DecoderBlock_UNet(in_channels=64, out_channels=32, skip_channels=0)
        self.block5 = DecoderBlock_UNet(in_channels=32, out_channels=16, skip_channels=0)

        # Final Conv: 16 channels -> 1 channel (ECG Signal)
        self.final_conv = nn.Conv1d(16, 1, kernel_size=1)
        self.refine_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, padding=7), # Kernel lớn để bao quát ngữ cảnh
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=1) # Trả về 1 kênh duy nhất
        )

    def forward(self, z, features_list):
        # features_list [f1, f2, f3] 
        # f1: low-level, f3: high-level 
        f1, f2, f3 = features_list 
        
        x = self.adapter(z) 
        x = self.block1(x, skip=f3) 
        x = self.block2(x, skip=f2) 
        x = self.block3(x, skip=f1) 
        
        x = self.block4(x) 
        x = self.block5(x) 
        
        x = self.final_conv(x)
        if x.shape[-1] != self.target_length:
            x = nn.functional.interpolate(
                x, 
                size=self.target_length, 
                mode='linear', 
                align_corners=False
            )
            x = self.refine_conv(x)
        return x

