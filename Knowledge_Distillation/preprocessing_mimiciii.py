import os
import random
import numpy as np
import scipy.io as scio
from typing import Dict, Any, Tuple

from preprocessing_signal import SignalProcessor, DataVisualizer
from utils import calculate_bsqi, calculate_sq_mask

# ==========================================
# CONSTANTS
# ==========================================
SEED = 44
PAUSE_TIME = 0.001
FS = 125  
WINDOW_SECONDS = 10
STEP_SIZE = 10 
WINDOW_SAMPLES = int(FS * WINDOW_SECONDS)

SLICE_LENGTH = 2400
TRAIN_STEP = 1200 
TEST_STEP = 2400   
TRAIN_RATIO = 0.8 
OVERLAP = 2400

# ==========================================
# COMMON FUNCTIONS
# ==========================================
def set_seed(seed_value: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    print(f"Random seed set to: {seed_value}")

def mask_filter(ecg, ppg): 
    sq_mask_array = calculate_sq_mask(ppg, fs=FS)
    ppg_sqi = np.mean(sq_mask_array)
    ecg_sqi = calculate_bsqi(ecg, fs=FS)
    if ppg_sqi < 0.3 or ecg_sqi < 0.3:
        return True
    return False

# ==========================================
# APPROACH 1: Split Continuous Signals THEN Slice 
# ==========================================
def slice_signal(ecg_sig: np.ndarray, ppg_sig: np.ndarray, record_name: str, step_size: int, stats: dict) -> Tuple[list, list, list]:
    ecgs, ppgs, names = [], [], []
    start_idx = 0
    min_len = min(len(ecg_sig), len(ppg_sig))
    
    while start_idx + SLICE_LENGTH <= min_len:
        stats['total_raw'] += 1
        ppg_slice = ppg_sig[start_idx:start_idx + SLICE_LENGTH]
        ecg_slice = ecg_sig[start_idx:start_idx + SLICE_LENGTH]
        
        if np.isnan(ppg_slice).any() or np.isnan(ecg_slice).any():
            stats['nan_flat_removed'] += 1
            start_idx += step_size
            continue
            
        if mask_filter(ecg_slice, ppg_slice):
            stats['sqi_removed'] += 1
            start_idx += step_size
            continue
            
        ecgs.append(ecg_slice)
        ppgs.append(ppg_slice)
        names.append(record_name)
        stats['final_valid'] += 1
        
        start_idx += step_size
        
    return ppgs, ecgs, names

def process_and_split_signals():
    data = scio.loadmat('Records.mat')
    records = data['records']
    
    train_data = {"ppgs": [], "ecgs": [], "records": []}
    test_data = {"ppgs": [], "ecgs": [], "records": []}
    
    processor = SignalProcessor(fs=FS)
    
    stats = {
        'total_raw': 0,           
        'nan_flat_removed': 0,    
        'sqi_removed': 0,         
        'final_valid': 0          
    }
    
    print(f"---  Train ({TRAIN_RATIO*100}%) | Test ({(1-TRAIN_RATIO)*100}%) ---")
    
    for index in range(records.size):
        ecg = records[index, 0]['ecg_II'][:, 0]
        ppg = records[index, 0]['ppg'][:, 0]
        record_name = str(index)
     
        ppg_preprocessed = processor.preprocessing_PPG(ppg)
        ecg_preprocessed = processor.preprocessing_ECG(ecg)
        ppg_preprocessed = processor.align_signals_cross_correlation(ecg_preprocessed, ppg_preprocessed)[0]
      
        min_len = min(len(ppg_preprocessed), len(ecg_preprocessed))
        
        if min_len < SLICE_LENGTH:
            continue
            
        split_point = int(min_len * TRAIN_RATIO)
        
        train_ecg_continuous = ecg_preprocessed[:split_point]
        train_ppg_continuous = ppg_preprocessed[:split_point]
        
        test_ecg_continuous = ecg_preprocessed[split_point:]
        test_ppg_continuous = ppg_preprocessed[split_point:]
        
        tr_ppgs, tr_ecgs, tr_names = slice_signal(train_ecg_continuous, train_ppg_continuous, record_name, TRAIN_STEP, stats)
        train_data["ppgs"].extend(tr_ppgs)
        train_data["ecgs"].extend(tr_ecgs)
        train_data["records"].extend(tr_names)
        
        ts_ppgs, ts_ecgs, ts_names = slice_signal(test_ecg_continuous, test_ppg_continuous, record_name, TEST_STEP, stats)
        test_data["ppgs"].extend(ts_ppgs)
        test_data["ecgs"].extend(ts_ecgs)
        test_data["records"].extend(ts_names)

    print("\n" + "="*60)
    print(" PREPROCESSING SUMMARY REPORT ")
    print("="*60)
    print(f"1. Total initial segments:           {stats['total_raw']}")
    print(f"2. Removed due to NaN/Flat Line:     {stats['nan_flat_removed']}")
    print(f"3. Removed due to poor SQI quality:  {stats['sqi_removed']}")
    print("-" * 60)
    print(f"FINAL RESULTS:")
    print(f"   - Total Train segments:           {len(train_data['records'])}")
    print(f"   - Total Test segments:            {len(test_data['records'])}")
    print("="*60)
    
    return train_data, test_data

def save_datasets(train_data: Dict[str, Any], test_data: Dict[str, Any], save_prefix: str):
    save_dir = "processed_data"
    os.makedirs(save_dir, exist_ok=True)

    splits = [("train", train_data), ("test", test_data)]

    for split_name, data in splits:
        if len(data["records"]) == 0:
            print(f"⚠️ No data for {split_name.upper()}")
            continue
            
        save_path = os.path.join(save_dir, f"{save_prefix}_{split_name}.npz")
        np.savez(save_path, **data)
        print(f"→ Saved {split_name.upper()}: {len(data['records'])} samples at '{save_path}'")

    print("Completed!")

# ==========================================
# APPROACH 2: Slice All Signals THEN Split 
# ==========================================
def load_and_slice_all_signals():
    data = scio.loadmat('Records.mat')
    records = data['records']
    record_ppgs = []
    record_ecgs = []
    record_names = []
    visualizer = DataVisualizer(fs=FS, window_seconds=WINDOW_SECONDS, 
                                pause_time=PAUSE_TIME, step_size=STEP_SIZE)
    processor = SignalProcessor(fs=FS)
    stats = {
        'total_raw': 0,          
        'nan_flat_removed': 0,    
        'sqi_removed': 0,         
        'final_valid': 0        
    }
    
    for index in range(records.size):
        ecg = records[index, 0]['ecg_II'][:, 0]
        ppg = records[index, 0]['ppg'][:, 0]
        samples_to_plot = 10 * FS
        record_name = str(index)
        
        ppg_preprocessed = processor.preprocessing_PPG(ppg)
        ecg_preprocessed = processor.preprocessing_ECG(ecg)

        min_len = min(len(ppg_preprocessed), len(ecg_preprocessed))
        print("Min length of signals: ", min_len)
        if min_len < SLICE_LENGTH:
            continue
            
        start_idx = 0
    
        while start_idx + SLICE_LENGTH <= min_len:
            stats['total_raw'] += 1
            ppg_slice = ppg_preprocessed[start_idx:start_idx + SLICE_LENGTH]
            ecg_slice = ecg_preprocessed[start_idx:start_idx + SLICE_LENGTH]
          
            if np.isnan(ppg_slice).any() or np.isnan(ecg_slice).any():
                stats['nan_flat_removed'] += 1
                start_idx += OVERLAP
                continue
            if mask_filter(ecg_slice, ppg_slice):
                stats['sqi_removed'] += 1
                start_idx += OVERLAP
                print("Segment failed quality check, skipping.")
                continue
                
            record_ppgs.append(ppg_slice)
            record_ecgs.append(ecg_slice)
            record_names.append(record_name)
            stats['final_valid'] += 1
            start_idx += OVERLAP

    return record_ppgs, record_ecgs, record_names

def split_segments_and_save_by_record(total_data: Dict[str, Any], save_prefix: str, ratios: Tuple[float, float]):
    train_ratio, test_ratio = ratios
    train_indices = []
    test_indices = []

    unique_records = list(set(total_data["records"]))
    record_array = np.array(total_data["records"])

    print(f"--- Ratio train to test: ( {train_ratio}:{test_ratio}) ---")

    for rec_name in unique_records:
        indices = np.where(record_array == rec_name)[0]
        
        num_seg = len(indices)
        split_point = int(num_seg * train_ratio)

        if split_point == 0 and num_seg > 0:
            split_point = 1
        elif split_point == num_seg and num_seg > 1:
            split_point = num_seg - 1

        train_indices.extend(indices[:split_point])
        test_indices.extend(indices[split_point:])

    save_dir = "processed_data"
    os.makedirs(save_dir, exist_ok=True)

    splits = [("train", train_indices), ("test", test_indices)]

    for split_name, current_indices in splits:
        save_path = os.path.join(save_dir, f"{save_prefix}_{split_name}.npz")
        
        save_dict = {
            "ecgs": [total_data["ecgs"][i] for i in current_indices],
            "ppgs": [total_data["ppgs"][i] for i in current_indices],
            "records": [total_data["records"][i] for i in current_indices]
        }

        np.savez(save_path, **save_dict)
        print(f"→ Save {split_name.upper()}: {len(current_indices)} samples at '{save_path}'")

    print("Complete!")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    set_seed(SEED)
    
    train_dataset, test_dataset = process_and_split_signals()
    save_datasets(train_dataset, test_dataset, save_prefix="mimic3_overlapping")
    
    # ---------------------------------------------------------
    # record_ppgs, record_ecgs, record_names = load_and_slice_all_signals()
    # total_data = {
    #     "ppgs": record_ppgs,
    #     "ecgs": record_ecgs,
    #     "records": record_names,
    # }
    # split_segments_and_save_by_record(total_data, "mimic3", (0.8, 0.2))