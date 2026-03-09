import os
import numpy as np
import scipy.io as scio
import random
from typing import Dict, Any, Tuple, List
from preprocessing import SignalProcessor
from utils import calculate_bsqi, calculate_sq_mask

SEED = 44
FS = 125  
SLICE_LENGTH = 2400
OVERLAP = 2400
TRAIN_RATIO = 0.8

def set_seed(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)

def mask_filter(ecg: np.ndarray, ppg: np.ndarray) -> bool:
    sq_mask_array = calculate_sq_mask(ppg, fs=FS)
    ppg_sqi = np.mean(sq_mask_array)
    ecg_sqi = calculate_bsqi(ecg, fs=FS)
    return ppg_sqi < 0.3 or ecg_sqi < 0.3

def load_and_slice_single_record(target_index: int) -> Tuple[List, List, List]:
    data = scio.loadmat('Records.mat')
    records = data['records']
    
    if target_index >= records.size:
        print(f" {target_index} over the number of records available!")
        return [], [], []

    record_ppgs, record_ecgs, record_names = [], [], []
    processor = SignalProcessor(fs=FS)
    
    record_data = records[target_index, 0]
    ecg = record_data['ecg_II'][:, 0]
    ppg = record_data['ppg'][:, 0]
    record_name = f"record_{target_index}"
    
    print(f"--- Processing only Record: {record_name} ---")
    
    ppg_p = processor.preprocessing_PPG(ppg)
    ecg_p = processor.preprocessing_ECG(ecg)
    # ppg_p = processor.align_signals_cross_correlation(ecg_p, ppg_p)[0]

    min_len = min(len(ppg_p), len(ecg_p))
    start_idx = 0
    
    while start_idx + SLICE_LENGTH <= min_len:
        p_slice = ppg_p[start_idx:start_idx + SLICE_LENGTH]
        e_slice = ecg_p[start_idx:start_idx + SLICE_LENGTH]
        
        if not np.isnan(p_slice).any() and not np.isnan(e_slice).any():
            if not mask_filter(e_slice, p_slice):
                record_ppgs.append(p_slice)
                record_ecgs.append(e_slice)
                record_names.append(record_name)
        start_idx += OVERLAP

    print(f" Complete: Obtained {len(record_ppgs)} segments from this record.")
    return record_ppgs, record_ecgs, record_names



def split_and_save_single_subject(total_data: Dict[str, Any], ratio: float = 0.7):
    num_seg = len(total_data["ppgs"])
    if num_seg < 2:
        print("Not enough dataset to split Train/Test!")
        return

    split_point = int(num_seg * ratio)
    
    indices = list(range(num_seg))

    train_idx = indices[:split_point]
    test_idx = indices[split_point:]

    save_dir = "processed_data_single"
    os.makedirs(save_dir, exist_ok=True)
    rec_name = total_data["records"][0]

    for name, idx_list in [("train", train_idx), ("test", test_idx)]:
        save_path = os.path.join(save_dir, f"{rec_name}_{name}.npz")
        np.savez(save_path, 
                 ecgs=np.array([total_data["ecgs"][i] for i in idx_list]),
                 ppgs=np.array([total_data["ppgs"][i] for i in idx_list]),
                 records=np.array([total_data["records"][i] for i in idx_list]))
        print(f"Save at {name.upper()}: {len(idx_list)} samples.")

if __name__ == "__main__":
    set_seed(SEED)
    
    TARGET_RECORD = 30
    
    ppgs, ecgs, names = load_and_slice_single_record(TARGET_RECORD)
    
    if ppgs:
        data_dict = {"ppgs": ppgs, "ecgs": ecgs, "records": names}
        split_and_save_single_subject(data_dict, ratio=TRAIN_RATIO)