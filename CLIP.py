import torch
from torch import nn
import torch.nn.functional as F
import math
from ResNet50 import ResNet50_1D
from VisionTransformer import SignalTransformer

OUTPUT_EMBED_DIM = 128 
INPUT_LENGTH = 2400 

class ECGEssembleCLIP(nn.Module):
    def __init__(self, embed_dim=OUTPUT_EMBED_DIM):
        super(ECGEssembleCLIP, self).__init__()

        self.encode_ecg_resnet = ResNet50_1D(input_length=INPUT_LENGTH,
        layers=[3, 4, 6, 3] ,
        num_classes=OUTPUT_EMBED_DIM)

        self.encode_ecg_transformer =  SignalTransformer(input_length=INPUT_LENGTH,
        patch_size=40,
        width=512,
        layers=6,
        heads=8,
        output_dim=OUTPUT_EMBED_DIM)

        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def forward(self, ecg_original, ppg_original):

        ecg_original_features = self.encode_ecg_resnet(ecg_original)
        ecg_predicted_features = self.encode_ecg_transformer(ppg_original)

        ecg_original_features = ecg_original_features / ecg_original_features.norm(dim=1, keepdim=True)
        ecg_predicted_features = ecg_predicted_features / ecg_predicted_features.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        
       
        logits_per_original = logit_scale * ecg_original_features @ ecg_predicted_features.t()
        
        logits_per_predicted = logits_per_original.t()

        return logits_per_original, logits_per_predicted

