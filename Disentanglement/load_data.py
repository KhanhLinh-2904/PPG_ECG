from torch.utils.data import Dataset
import torch
import numpy as np
class LoadData(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        self.ppgs = data["ppgs"]
        self.ecgs = data["ecgs"]
        self.labels = data["labels"]
        self.record = data["records"]
        # self.fs = 128

    def __len__(self):
        return len(self.ppgs)

    def __getitem__(self, idx):
        ppg = torch.tensor(self.ppgs[idx], dtype=torch.float32)
        ecg = torch.tensor(self.ecgs[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)  
        record_name = str(self.record[idx])
        # return ecg, ppg, record_name, label
        # return ecg, ppg, record_name
        return  ppg, ecg, record_name, label
        # return  ecg, ecg
    
# import numpy as np
# import torch
# from torch.utils.data import Dataset

# class LoadData(Dataset):
#     def __init__(self, npz_path):
#         data = np.load(npz_path, allow_pickle=True)
#         self.ppgs = data["ppgs"]
#         self.ecgs = data["ecgs"]
#         self.record = data["records"]

#     def __len__(self):
#         return len(self.ppgs)

#     def minmax_scale(self, signal: np.ndarray) -> np.ndarray:
#         min_val = np.min(signal)
#         max_val = np.max(signal)
        
#         # tránh chia cho 0 (signal constant)
#         if max_val - min_val < 1e-8:
#             return np.zeros_like(signal)
        
#         # scale về [-1, 1]
#         scaled = 2 * (signal - min_val) / (max_val - min_val) - 1
#         return scaled

#     def __getitem__(self, idx):
#         ppg = self.minmax_scale(self.ppgs[idx])
#         ecg = self.minmax_scale(self.ecgs[idx])

#         ppg = torch.tensor(ppg, dtype=torch.float32)
#         ecg = torch.tensor(ecg, dtype=torch.float32)

#         return ppg, ecg

    