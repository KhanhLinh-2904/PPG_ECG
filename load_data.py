from torch.utils.data import Dataset
import torch
import numpy as np 
class LoadData(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        self.ppgs = data["ppgs"]
        self.ecgs = data["ecgs"]
        self.labels = data["labels"]
        self.group_id = data["groupIDs"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ppg = torch.tensor(self.ppgs[idx], dtype=torch.float32)
        ecg = torch.tensor(self.ecgs[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)  # classification target
        group_id = torch.tensor(self.group_id[idx], dtype=torch.long)  # group ID
        # return ecg, ppg, label
        return ecg, ppg, label, group_id
    
