import os
import numpy as np
import scipy.io as scio
from preprocessing import DataVisualizer, SignalProcessor
import random
from typing import Dict, Any, Tuple, List
from utils import calculate_bsqi, calculate_sq_mask

SEED = 44
PAUSE_TIME = 0.001
FS = 125  
# STEP_SIZE = 4

# Các thông số cắt segment
SLICE_LENGTH = 2400
TRAIN_STEP = 1200  # Bước nhảy cho Train (1200 < 2400 -> overlap 50%)
TEST_STEP = 2400   # Bước nhảy cho Test (2400 = SLICE_LENGTH -> KHÔNG overlap)
TRAIN_RATIO = 0.8  # Tỷ lệ chia tín hiệu đầu vào

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

def slice_signal(ecg_sig: np.ndarray, ppg_sig: np.ndarray, record_name: str, step_size: int, stats: dict) -> Tuple[list, list, list]:
    """Hàm phụ trợ để cắt segment cho một đoạn tín hiệu cho trước với step_size cụ thể."""
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
            # print("Segment failed quality check, skipping.")
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
    
    # Khởi tạo data dictionary cho Train và Test
    train_data = {"ppgs": [], "ecgs": [], "records": []}
    test_data = {"ppgs": [], "ecgs": [], "records": []}
    
    processor = SignalProcessor(fs=FS)
    
    stats = {
        'total_raw': 0,           
        'nan_flat_removed': 0,    
        'sqi_removed': 0,         
        'final_valid': 0          
    }
    
    print(f"--- Đang xử lý từng bản ghi: Train ({TRAIN_RATIO*100}%) | Test ({(1-TRAIN_RATIO)*100}%) ---")
    
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
            
        # Chia tín hiệu liên tục thành 80% train và 20% test
        split_point = int(min_len * TRAIN_RATIO)
        
        # 80% cho Train
        train_ecg_continuous = ecg_preprocessed[:split_point]
        train_ppg_continuous = ppg_preprocessed[:split_point]
        
        # 20% cho Test
        test_ecg_continuous = ecg_preprocessed[split_point:]
        test_ppg_continuous = ppg_preprocessed[split_point:]
        
        # Cắt segment cho tập Train VỚI OVERLAP (step_size = TRAIN_STEP)
        tr_ppgs, tr_ecgs, tr_names = slice_signal(train_ecg_continuous, train_ppg_continuous, record_name, TRAIN_STEP, stats)
        train_data["ppgs"].extend(tr_ppgs)
        train_data["ecgs"].extend(tr_ecgs)
        train_data["records"].extend(tr_names)
        
        # Cắt segment cho tập Test KHÔNG OVERLAP (step_size = TEST_STEP)
        ts_ppgs, ts_ecgs, ts_names = slice_signal(test_ecg_continuous, test_ppg_continuous, record_name, TEST_STEP, stats)
        test_data["ppgs"].extend(ts_ppgs)
        test_data["ecgs"].extend(ts_ecgs)
        test_data["records"].extend(ts_names)

    # In báo cáo tổng hợp
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
    """Lưu trữ dữ liệu đã chia sẵn vào thư mục."""
    save_dir = "processed_data"
    os.makedirs(save_dir, exist_ok=True)

    splits = [("train", train_data), ("test", test_data)]

    for split_name, data in splits:
        if len(data["records"]) == 0:
            print(f"⚠️ Không có dữ liệu để lưu cho tập {split_name.upper()}")
            continue
            
        save_path = os.path.join(save_dir, f"{save_prefix}_{split_name}.npz")
        np.savez(save_path, **data)
        print(f"→ Saved {split_name.upper()}: {len(data['records'])} samples at '{save_path}'")

    print("Hoàn tất!")

if __name__ == "__main__":
    set_seed(SEED)
    
    # Bước 1: Tiền xử lý, chia 80/20 tín hiệu gốc, và cắt window (overlap cho train, no-overlap cho test)
    train_dataset, test_dataset = process_and_split_signals()
    
    # Bước 2: Lưu lại thành file .npz
    save_datasets(train_dataset, test_dataset, save_prefix="mimic3_signal_split")