from torch.utils.data import Dataset
import torch
import numpy as np 
class LoadDataPPG(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        self.ppgs = data["x"]
        self.labels = data["y"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ppg = torch.tensor(self.ppgs[idx], dtype=torch.float32)
        ppg = ppg.squeeze(-1) 
        ppg_tripled = ppg.repeat(3)
        label = torch.tensor(self.labels[idx], dtype=torch.long)  # classification target

        return ppg_tripled, label
