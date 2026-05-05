import numpy as np
from typing import List, Tuple, Dict, Any
import os
import wfdb
import random
from collections import defaultdict
from utils import calculate_bsqi, calculate_sq_mask
from preprocessing_signal import SignalProcessor, DataVisualizer
# --- (CONSTANTS) ---
SEED = 44
PAUSE_TIME = 0.001
FS = 125  
WINDOW_SECONDS = 10
STEP_SIZE = 10 
WINDOW_SAMPLES = int(FS * WINDOW_SECONDS)
SLICE_LENGTH = 2400
OVERLAP = 2400
THRESHOLD_SIMILARITY = 0.9

def set_seed(seed_value: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    print(f"Random seed set to: {seed_value}")
# ---  (SIGNAL PROCESSING CLASS) ---

def mask_filter(ecg, ppg): 
    sq_mask_array = calculate_sq_mask(ppg, fs=125)
    ppg_sqi = np.mean(sq_mask_array)
    ecg_sqi = calculate_bsqi(ecg, fs=125)
    if ppg_sqi < 0.3 or ecg_sqi < 0.3:
        return True
    return False


def get_all_records(datapath: str) -> List[str]:
    if not os.path.exists(datapath):
        return []
    records = [f.replace('.hea', '') for f in os.listdir(datapath) if f.endswith('.hea')]
    if not records:
        records = [f.split('.')[0] for f in os.listdir(datapath) if f.endswith('.dat')]
    return sorted(list(set(records)))


def load_and_slice_all_signals(datapath: str, record_list: List[str]):
    record_ppgs = []
    record_ecgs = []
    record_names = []

    stats = {
        'total_slices': 0,
        'nan_filtered': 0,
        'sqi_filtered': 0
    }
    
    processor = SignalProcessor(fs=FS)
    visualizer = DataVisualizer(fs=FS, window_seconds=WINDOW_SECONDS, 
                                pause_time=PAUSE_TIME, step_size=STEP_SIZE)
    
    for i, record_name in enumerate(record_list):
        print(f"\n===  {i+1}/{len(record_list)}: {record_name} ===")
        record = wfdb.rdrecord(os.path.join(datapath, record_name))
        signal_data = record.p_signal
        if signal_data is None or signal_data.shape[1] < 2:
            print(f"error {record_name} not enough PPG ECG")
            continue
            
        ppg = signal_data[:, 0]
        ecg = signal_data[:, 1]
        samples_to_plot = 10 * FS
        # visualizer.visualize_sliding_record(record_name, ppg, ecg)
        # visualizer.visualize_specific_segment(record_name, ppg[:samples_to_plot], ecg[:samples_to_plot])
        ppg_preprocessed = processor.preprocessing_PPG(ppg)
        ecg_preprocessed = processor.preprocessing_ECG(ecg)
        # visualizer.visualize_sliding_record(record_name, ppg_preprocessed, ecg_preprocessed)
        # visualizer.visualize_specific_segment(record_name, ppg_preprocessed[:samples_to_plot], ecg_preprocessed[:samples_to_plot])

        # Aligment PPG and ECG
        ppg_preprocessed = processor.align_signals_cross_correlation(ecg_preprocessed, ppg_preprocessed)[0]
        # visualizer.visualize_sliding_record(record_name, ppg_preprocessed, ecg_preprocessed)
        # visualizer.visualize_sliding_record(record_name, ppg_preprocessed, ppg)
        # visualizer.visualize_specific_segment(record_name, ppg_preprocessed[:samples_to_plot], ecg_preprocessed[:samples_to_plot])



        min_len = min(len(ppg_preprocessed), len(ecg_preprocessed))
        print("Min length of signals: ", min_len)
        if min_len < SLICE_LENGTH:
            continue
        start_idx = 0
        while start_idx + SLICE_LENGTH <= min_len:
            stats['total_slices'] += 1
            ppg_slice = ppg_preprocessed[start_idx:start_idx + SLICE_LENGTH]
            ecg_slice = ecg_preprocessed[start_idx:start_idx + SLICE_LENGTH]
             
           
            if np.isnan(ppg_slice).any() or np.isnan(ecg_slice).any():
                stats['nan_filtered'] += 1
                start_idx += OVERLAP
                print("Found NaN values, skipping this segment.")
                continue
            if mask_filter(ecg_slice, ppg_slice):
                stats['sqi_filtered'] += 1
                start_idx += OVERLAP
                print("Segment failed quality check, skipping.")
                continue
            record_ppgs.append(ppg_slice)
            record_ecgs.append(ecg_slice)
            record_names.append(record_name)
            start_idx += OVERLAP
        print("Number of segments extracted from this record: ", len(record_ppgs))
    print("Total number of PPG segments after slicing: ", len(record_ppgs))
    return record_ppgs, record_ecgs, record_names, stats

def split_segments_and_save_by_record(total_data: Dict[str, Any], save_prefix: str, ratios: Tuple[float, float], seed: int = 42):
    os.makedirs("datasets", exist_ok=True)
    np.random.seed(seed)
    random.seed(seed)

    all_records = np.array(total_data["records"])
    all_labels = np.array(total_data["labels"])
    
    record_to_label = {}
    unique_records = np.unique(all_records)
    
    for rec in unique_records:
        rec_labels = all_labels[all_records == rec]
        most_frequent_label = np.bincount(rec_labels).argmax()
        record_to_label[rec] = most_frequent_label

    af_records = [r for r, l in record_to_label.items() if l == 1]
    non_af_records = [r for r, l in record_to_label.items() if l == 0]

    np.random.shuffle(af_records)
    np.random.shuffle(non_af_records)

    def get_split_names(rec_list, ratio):
        n_train = int(len(rec_list) * ratio)
        return rec_list[:n_train], rec_list[n_train:]

    train_af, test_af = get_split_names(af_records, ratios[0])
    train_non, test_non = get_split_names(non_af_records, ratios[0])

    train_record_names = train_af + train_non
    test_record_names = test_af + test_non

    raw_splits = {
        "train": np.where(np.isin(all_records, train_record_names))[0].tolist(),
        "test": np.where(np.isin(all_records, test_record_names))[0].tolist()
    }
    for split_name, indices in raw_splits.items():
        split_labels = all_labels[indices]
        af_idx = [idx for idx in indices if all_labels[idx] == 1]
        non_af_idx = [idx for idx in indices if all_labels[idx] == 0]
        
        target = min(len(af_idx), len(non_af_idx))
        
        if target == 0:
            print(f" {split_name} is not enough to balance!")
            continue

        if len(af_idx) > len(non_af_idx):
            final_af = inverse_proportional_sampling(indices, target, all_labels, all_records, 1)
            final_non_af = non_af_idx
        else:
            final_non_af = inverse_proportional_sampling(indices, target, all_labels, all_records, 0)
            final_af = af_idx

        final_indices = final_af + final_non_af
        np.random.shuffle(final_indices)

        save_path = f"datasets/{save_prefix}_{split_name}.npz"
        save_dict = {
            "ecgs": np.array([total_data["ecgs"][i] for i in final_indices]),
            "labels": np.array([total_data["labels"][i] for i in final_indices]),
            "records": np.array([total_data["records"][i] for i in final_indices])
        }
        if "ppgs" in total_data:
            save_dict["ppgs"] = np.array([total_data["ppgs"][i] for i in final_indices])

        np.savez(save_path, **save_dict)
        print(f"→ Save {split_name.upper()}: {len(final_indices)} segments (AF: {len(final_af)}, Non-AF: {len(final_non_af)})")

def inverse_proportional_sampling(indices, target_count, records, label_indices):
   
    record_groups = defaultdict(list)
    for idx in label_indices:
        record_groups[records[idx]].append(idx)
    
    rec_ids = list(record_groups.keys())
    counts = np.array([len(record_groups[rid]) for rid in rec_ids])
    
    counts = np.maximum(counts, 1)

    inverse_counts = 1.0 / counts
    weights = inverse_counts / np.sum(inverse_counts)
    
    keep_counts = np.round(weights * target_count).astype(int)
    
    final_keep_indices = []
    
    for i, rid in enumerate(rec_ids):
        available = record_groups[rid]
        n_to_pick = min(len(available), keep_counts[i])
        
        if n_to_pick > 0:
            chosen = np.random.choice(available, n_to_pick, replace=False).tolist()
            final_keep_indices.extend(chosen)

    current_total = len(final_keep_indices)
    
    if current_total < target_count:
        remaining_needed = target_count - current_total
        all_potential = [idx for rid in rec_ids for idx in record_groups[rid]]
        leftovers = list(set(all_potential) - set(final_keep_indices))
        
        if leftovers:
            extra = np.random.choice(leftovers, min(len(leftovers), remaining_needed), replace=False).tolist()
            final_keep_indices.extend(extra)
            
    elif current_total > target_count:
        excess = current_total - target_count
        final_keep_indices = np.random.choice(final_keep_indices, target_count, replace=False).tolist()

    return final_keep_indices

def split_segments_and_save(total_data: Dict[str, Any], save_prefix: str, ratios: Tuple[float, float] = (0.8, 0.2), seed: int = 42):
    os.makedirs("datasets", exist_ok=True)
    np.random.seed(seed)
    
    all_labels = np.array(total_data["labels"])
    all_records = np.array(total_data["records"])

    record_map = defaultdict(list)
    for i, record_id in enumerate(all_records):
        record_map[record_id].append(i)
    
    raw_indices = {"train": [], "test": []}
    for record_id, indices in record_map.items():
        indices.sort()
        split_point = int(len(indices) * ratios[0])
        raw_indices["train"].extend(indices[:split_point])
        raw_indices["test"].extend(indices[split_point:])

    #
    for split_name, current_indices in raw_indices.items():
        af_idx = [idx for idx in current_indices if all_labels[idx] == 1]
        non_af_idx = [idx for idx in current_indices if all_labels[idx] == 0]
        
        target = min(len(af_idx), len(non_af_idx))
        
        if target == 0:
            print(f"⚠️ Tập {split_name} không đủ dữ liệu cả 2 lớp!")
            continue

     
        if len(af_idx) > len(non_af_idx):
            print(f"Undersampling AF class from {len(af_idx)} to {len(non_af_idx)}...")
            target = len(non_af_idx)
            final_af = inverse_proportional_sampling(current_indices, target, all_records, af_idx)
            final_non_af = non_af_idx
        else:
            print(f"Undersampling Non-AF class from {len(non_af_idx)} to {len(af_idx)}...")
            target = len(af_idx)
            final_non_af = inverse_proportional_sampling(current_indices, target, all_records, non_af_idx)
            final_af = af_idx

        final_combined = final_af + final_non_af
        np.random.shuffle(final_combined)

        save_path = f"datasets/{save_prefix}_{split_name}.npz"
        save_dict = {
            "ecgs": np.array([total_data["ecgs"][i] for i in final_combined]),
            "labels": np.array([total_data["labels"][i] for i in final_combined]),
            "records": np.array([total_data["records"][i] for i in final_combined])
        }
        
        if "ppgs" in total_data:
            save_dict["ppgs"] = np.array([total_data["ppgs"][i] for i in final_combined])

        np.savez(save_path, **save_dict)

        print(f" {split_name.upper()}: {len(final_combined)} samples (AF: {len(final_af)}, Non-AF: {len(final_non_af)})")
        print(f"   Saved at '{save_path}'")

def save_data(total_data: Dict[str, Any], filename: str):
   
    os.makedirs("datasets", exist_ok=True)
    np.random.seed(SEED)
    
    all_labels = np.array(total_data["labels"]).flatten()
    all_records = np.array(total_data["records"]).flatten()
    all_ecgs = np.array(total_data["ecgs"])
    
    current_indices = np.arange(len(all_labels))
    
    af_idx = np.where(all_labels == 1)[0]
    non_af_idx = np.where(all_labels == 0)[0]
    
    target = min(len(af_idx), len(non_af_idx))
    
    if target == 0:
        print(f" Dataset is not enough balanced! (AF: {len(af_idx)}, Non-AF: {len(non_af_idx)})")
        return

    print(f"--- Balance for dataset in file '{filename}' ---")

    if len(af_idx) > len(non_af_idx):
        print(f"Undersampling AF class from {len(af_idx)} to {len(non_af_idx)}...")
        final_af = inverse_proportional_sampling(current_indices, target, all_records, af_idx)
        final_non_af = non_af_idx
    else:
        print(f"Undersampling Non-AF class from {len(non_af_idx)} to {len(af_idx)}...")
        final_non_af = inverse_proportional_sampling(current_indices, target, all_records, non_af_idx)
        final_af = af_idx

    final_combined = np.concatenate([final_af, final_non_af]).astype(int)
    np.random.shuffle(final_combined)

    save_dict = {
        "ecgs": all_ecgs[final_combined],
        "labels": all_labels[final_combined],
        "records": all_records[final_combined]
    }
    
    if "ppgs" in total_data:
        all_ppgs = np.array(total_data["ppgs"])
        save_dict["ppgs"] = all_ppgs[final_combined]

    save_path = os.path.join("datasets", filename)
    if not save_path.endswith('.npz'):
        save_path += '.npz'
        
    np.savez_compressed(save_path, **save_dict)

    print(f" Completed:  {len(final_combined)}  ( 1:1 -> AF: {len(final_af)}, Non-AF: {len(final_non_af)})")
    print(f"   Data saved at: '{save_path}'")

if __name__ == "__main__":
    set_seed(SEED)
    datapath_non_af = "/home/linhhima/Pre_processing_data/Datasets/mimic_perform_non_af_wfdb" 
    datapath_af = "/home/linhhima/Pre_processing_data/Datasets/mimic_perform_af_wfdb"

    records_sample_non_af = ['mimic_perform_non_af_001', 'mimic_perform_non_af_002',
                    'mimic_perform_non_af_003', 
                    'mimic_perform_non_af_005',
                    'mimic_perform_non_af_007', 'mimic_perform_non_af_008',
                    'mimic_perform_non_af_009', 
                    'mimic_perform_non_af_011', 
                    'mimic_perform_non_af_013', 
                     'mimic_perform_non_af_016']

    records_sample_af = ['mimic_perform_af_001', 'mimic_perform_af_002',
                    'mimic_perform_af_003', 'mimic_perform_af_004',
                     'mimic_perform_af_012',
                    'mimic_perform_af_014',
                    'mimic_perform_af_015', 'mimic_perform_af_016',
                     'mimic_perform_af_018',
                    'mimic_perform_af_019'
                    ]
    non_af_data_ppg, non_af_data_ecg, non_af_records, stats_non_af = load_and_slice_all_signals(datapath_non_af, records_sample_non_af)
    af_data_ppg, af_data_ecg, af_records, stats_af = load_and_slice_all_signals(datapath_af, records_sample_af)
    # --- PRINT DETAILED STATISTICAL REPORT ---
    print("\n" + "="*60)
    print(" DATA FILTERING STATISTICS (DROPOUT REPORT)")
    print("="*60)
    
    print(f"{'Category':<35} | {'AF Label':<10} | {'Non-AF Label':<12}")
    print("-" * 65)
    print(f"{'1. Total Initial Segments':<35} | {stats_af['total_slices']:<10} | {stats_non_af['total_slices']:<12}")
    
    print(f"{'2. Filtered by NaN/Flat signal':<35} | {stats_af['nan_filtered']:<10} | {stats_non_af['nan_filtered']:<12}")
    print(f"{'   Dropout Rate (%)':<35} | {stats_af['nan_filtered']/max(1,stats_af['total_slices'])*100:>9.2f}% | {stats_non_af['nan_filtered']/max(1,stats_non_af['total_slices'])*100:>11.2f}%")
    
    print(f"{'3. Filtered by SQI (masking)':<35} | {stats_af['sqi_filtered']:<10} | {stats_non_af['sqi_filtered']:<12}")
    print(f"{'   Dropout Rate (%)':<35} | {stats_af['sqi_filtered']/max(1,stats_af['total_slices'])*100:>9.2f}% | {stats_non_af['sqi_filtered']/max(1,stats_non_af['total_slices'])*100:>11.2f}%")
    non_af_labels = [0] * len(non_af_data_ppg)
    af_labels = [1] * len(af_data_ppg)

    
    all_ppgs = non_af_data_ppg + af_data_ppg
    all_ecgs = non_af_data_ecg + af_data_ecg
    all_records = non_af_records + af_records
    all_labels = non_af_labels + af_labels

    total_data = {
        "ppgs": all_ppgs,
        "ecgs": all_ecgs,
        "labels": all_labels,    
        "records": all_records        
    }

    print(f"Total number of Non-AF segments: {len(non_af_data_ppg)}")
    print(f"Total number of AF segments: {len(af_data_ppg)}")
    print(f"Total data after merging: {len(total_data['ppgs'])}")
    save_data(total_data, "total_mimic_af_min_max.npz")
    # split_segments_and_save(total_data, save_prefix="normal_remove24", ratios=(0.8, 0.2))
    # split_segments_and_save_by_record(total_data, save_prefix="record", ratios=(0.8, 0.2))
    # split_segments_and_save(total_data, save_prefix="total_z", ratios=(0.8, 0.2))
    # split_segments_and_save_by_record(total_data, save_prefix="total_record_mm", ratios=(0.8, 0.2))

