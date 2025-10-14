# import numpy as np

# data_path  = "datasets/deepbeat_data_extract/val_cleaned_signals.npz"
# data = np.load(data_path)
# labels = data["y"]
# # quality = data[""]
# # print("Quality shape: ",np.any(quality==2))
# print("Length of dataset: ", len(labels))
# norm = 0
# af = 0 
# for i in labels:
#     if i == 0:
#         norm += 1
#     else:
#         af += 1
# print("Non_af: ", norm)
# print("AF: ", af)

# test.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from load_data_ecg import LoadData
from model import  Res34SimSiamNoise
AE_CHECKPOINT = "checkpoint_epoch_192.pth"
DIM1, DIM2 = 512, 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "saved_models/model_21.pt"

def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Remove 'module.' prefix if present
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        new_key = k.replace("module.", "")  # strip "module."
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict, strict=False)
    return model
# === Load model ===
model = Res34SimSiamNoise(DIM1, DIM2, single_source_mode=True)
model = load_checkpoint(model, checkpoint_path)
model.eval()

# === Print out model structure ===
print("\n=== Model Structure ===")
print(model)

# === Access predictors and projectors ===
encoder1_params = list(model.encoder1.parameters())
encoder2_params = list(model.encoder2.parameters())
projector1_params = list(model.projector1.parameters())
projector2_params = list(model.projector2.parameters())
predictor1_params = list(model.predictor1.parameters())
predictor2_params = list(model.predictor2.parameters())

# print(f"\nPredictor1 layers: {len(predictor1_params)}")
# print(f"Predictor2 layers: {len(predictor2_params)}")
# print(f"Projector1 layers: {len(projector1_params)}")
# print(f"Projector2 layers: {len(projector2_params)}")

# # === Optional: print specific weight shapes ===
# print("\n--- Example Parameter Shapes ---")
# for name, param in model.named_parameters():
#     if any(k in name for k in ["predictor", "projector"]):
#         print(f"{name:<40} {tuple(param.shape)}")

# === Compute similarity between predictor/projector pairs ===
def cosine_similarity_params(params_a, params_b):
    """
    Compute cosine similarity between flattened parameter tensors of two modules.
    """
    vec_a = torch.cat([p.flatten() for p in params_a]).detach()
    vec_b = torch.cat([p.flatten() for p in params_b]).detach()
    return F.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0)).item()

sim_pred = cosine_similarity_params(encoder1_params, encoder2_params)
sim_proj = cosine_similarity_params(projector1_params, projector2_params)

print("\n=== Parameter Similarity ===")
print(f"Encoder1 ↔ Encoder2 cosine similarity: {sim_pred:.6f}")
print(f"Projector1 ↔ Projector2 cosine similarity: {sim_proj:.6f}")
