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
WINDOW_SECONDS = 10
STEP_SIZE = 10 
WINDOW_SAMPLES = int(FS * WINDOW_SECONDS)
SLICE_LENGTH = 2400
OVERLAP = 2400
def set_seed(seed_value: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    print(f"Random seed set to: {seed_value}")

def mask_filter(ecg, ppg): 
    sq_mask_array = calculate_sq_mask(ppg, fs=125)
    ppg_sqi = np.mean(sq_mask_array)
    ecg_sqi = calculate_bsqi(ecg, fs=125)
    if ppg_sqi < 0.3 or ecg_sqi < 0.3:
        return True
    return False

def load_and_slice_all_signals():
    data = scio.loadmat('Records.mat')
    records = data['records']
    Fs = 125
    record_ppgs = []
    record_ecgs = []
    record_names = []
    visualizer = DataVisualizer(fs=Fs, window_seconds=WINDOW_SECONDS, 
                                pause_time=PAUSE_TIME, step_size=STEP_SIZE)
    processor = SignalProcessor(fs=FS)
    stats = {
    'total_raw': 0,           # Total initial segments extracted
    'nan_flat_removed': 0,    # Removed due to NaN or flatline signals
    'sqi_removed': 0,         # Removed due to poor signal quality (SQI)
    'final_valid': 0          # Final count of valid segments
    }
    for index in range(records.size):
        ecg = records[index, 0]['ecg_II'][:, 0]
        ppg = records[index, 0]['ppg'][:, 0]
        samples_to_plot = 10 * Fs
        record_name = str(index)
        # visualizer.visualize_sliding_record(record_name, ppg, ecg)
        # visualizer.visualize_specific_segment(record_name, ppg[:samples_to_plot], ecg[:samples_to_plot])
        ppg_preprocessed = processor.preprocessing_PPG(ppg)
        ecg_preprocessed = processor.preprocessing_ECG(ecg)
        # visualizer.visualize_sliding_record(record_name, ppg_preprocessed, ecg_preprocessed)
        # visualizer.visualize_specific_segment(record_name, ppg_preprocessed[:samples_to_plot], ecg_preprocessed[:samples_to_plot])

        # # Aligment PPG and ECG
        # ppg_preprocessed = processor.align_signals_cross_correlation(ecg_preprocessed, ppg_preprocessed)[0]
        # visualizer.visualize_sliding_record(record_name, ppg_preprocessed, ecg_preprocessed)
        # visualizer.visualize_sliding_record(record_name, ppg_preprocessed, ppg)
        # visualizer.visualize_specific_segment(record_name, ppg_preprocessed[:samples_to_plot], ecg_preprocessed[:samples_to_plot])

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

    # # --- PRINT PROCESSING SUMMARY ---
    # print("\n" + "="*60)
    # print(" PREPROCESSING SUMMARY REPORT ")
    # print("="*60)
    # print(f"1. Total initial segments:           {stats['total_raw']}")
    # print(f"2. Removed due to NaN/Flat Line:     {stats['nan_flat_removed']}")
    # print(f"3. Removed due to poor SQI quality:  {stats['sqi_removed']}")
    # print("-" * 60)
    # print(f"FINAL RESULTS:")
    # print(f"   - Total valid samples:            {stats['final_valid']}")
    # print(f"   - Retention Rate:                 {stats['final_valid']/max(1,stats['total_raw']):.2%}")
    # print("="*60)
    # print("Total PPG segments after slicing: ", len(record_ppgs))
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

if __name__ == "__main__":
    set_seed(SEED)
    record_ppgs, record_ecgs, record_names = load_and_slice_all_signals()
    total_data = {
        "ppgs": record_ppgs,
        "ecgs": record_ecgs,
        "records": record_names,
    }
    split_segments_and_save_by_record(total_data, "mimic3_overlapping", (0.8, 0.2))
