from torch.utils.data import Dataset
import torch
import numpy as np
from utils import calculate_bsqi, calculate_ppg_sqi, calculate_sq_mask
threshold_ppg_sqi = 0.3
threshold_ecg_sqi = 0.5
class LoadData(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        self.ppgs = data["ppgs"]
        self.ecgs = data["ecgs"]
        # self.labels = data["labels"]
        self.record = data["records"]
        self.fs = 128

    def __len__(self):
        return len(self.record)

    def __getitem__(self, idx):
        ppg = torch.tensor(self.ppgs[idx], dtype=torch.float32)
        ecg = torch.tensor(self.ecgs[idx], dtype=torch.float32)
        # label = torch.tensor(self.labels[idx], dtype=torch.long)  
        record_name = str(self.record[idx])
        # return ecg, ppg, record_name, label
        return ecg, ppg, record_name
    
class PPG2ECG_Dataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        self.ppgs = data["ppgs"]
        self.ecgs = data["ecgs"]
        self.record = data["records"]
        self.fs = 128

    def __len__(self):
        return len(self.record)

    def __getitem__(self, idx):
        sq_mask_array = calculate_sq_mask(self.ppgs[idx], fs=self.fs)
        
        ppg_sqi = np.mean(sq_mask_array)
        
        ecg_sqi = calculate_bsqi(self.ecgs[idx], fs=self.fs)
        
        ppg = torch.tensor(self.ppgs[idx], dtype=torch.float32)
        ecg = torch.tensor(self.ecgs[idx], dtype=torch.float32)
        ppg_mask = torch.tensor(sq_mask_array, dtype=torch.float32) 
        
        record_name = str(self.record[idx])
        return ecg, ppg, record_name, ppg_mask, ppg_sqi, ecg_sqi 


def quality_aware_collate_fn(batch):
    valid_ecg = []
    valid_ppg = []
    valid_records = []
    valid_ppg_mask = [] 
    valid_ppg_sqi = []
    valid_ecg_sqi = []
    
    for item in batch:
        ecg, ppg, record_name, ppg_mask, ppg_sqi, ecg_sqi = item
        
        if float(ecg_sqi) < threshold_ecg_sqi: 
            continue
            
        if float(ppg_sqi) < threshold_ppg_sqi:
            continue
            
        valid_ecg.append(ecg)
        valid_ppg.append(ppg)
        valid_records.append(record_name)
        valid_ppg_mask.append(ppg_mask) 
        valid_ppg_sqi.append(ppg_sqi)
        valid_ecg_sqi.append(ecg_sqi)
        
    if len(valid_ecg) == 0:
        return None 
        
    batched_ecg = torch.stack(valid_ecg)
    batched_ppg = torch.stack(valid_ppg)
    batched_ppg_mask = torch.stack(valid_ppg_mask) 
    
    batched_ppg_sqi = torch.tensor(valid_ppg_sqi, dtype=torch.float32)
    batched_ecg_sqi = torch.tensor(valid_ecg_sqi, dtype=torch.float32)
    return batched_ecg, batched_ppg, valid_records, batched_ppg_mask, batched_ppg_sqi, batched_ecg_sqi


class PPG2ECG_BUT_Dataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        self.ppgs = data["ppgs"]
        self.ecgs = data["ecgs"]
        self.record = data["records"]
        self.fs_ECG = 1000
        self.fs_PPG = 30

    def __len__(self):
        return len(self.record)

    def __getitem__(self, idx):
        sq_mask_array = calculate_sq_mask(self.ppgs[idx], fs=self.fs_PPG)

        ppg_sqi = np.mean(sq_mask_array)

        ecg_sqi = calculate_bsqi(self.ecgs[idx], fs=self.fs_ECG)

        ppg = torch.tensor(self.ppgs[idx], dtype=torch.float32)
        ecg = torch.tensor(self.ecgs[idx], dtype=torch.float32)
        ppg_mask = torch.tensor(sq_mask_array, dtype=torch.float32) 
        
        record_name = str(self.record[idx])
        return ecg, ppg, record_name, ppg_mask, ppg_sqi, ecg_sqi 
