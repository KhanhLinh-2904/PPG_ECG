from torch.utils.data import Dataset
import torch
import numpy as np 
class LoadDataECG(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        self.ecgs = data["ecgs"]
        self.labels = data["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        self.ecgs[idx] = 1 - self.ecgs[idx]
        ecg = torch.tensor(self.ecgs[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)  # classification target

        return ecg, label
