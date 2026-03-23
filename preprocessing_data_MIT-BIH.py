import wfdb
import numpy as np
import os
import matplotlib.pyplot as plt
from preprocessing import SignalProcessor
import random
from utils import calculate_bsqi
TARGET_FS = 250
SLICE_LENGTH = 2400
signal_preprocess_ecg = SignalProcessor(fs=TARGET_FS)
seed = 42
def set_seed(seed_value=42):
    np.random.seed(seed_value)
    random.seed(seed_value)

set_seed(seed)

def mask_filter(ecg): 
    ecg_sqi = calculate_bsqi(ecg, fs=250)
    # print("ECG SQI: ", ecg_sqi)
    if ecg_sqi < 0.3:
        return True
    return False

def visualize_af_vs_normal_separate(segments, labels, records, fs):
    af_indices = np.where(labels == 1)[0]
    normal_indices = np.where(labels == 0)[0]
    if len(af_indices) == 0 or len(normal_indices) == 0:
        print("Warning: Not enough data of both types (AF and Normal) for comparison.")
        return

    selected_indices = [af_indices[0], normal_indices[0]]
    time = np.arange(segments.shape[1]) / fs
    
    for idx in selected_indices:
        plt.figure(figsize=(15, 5))
        
        signal = segments[idx]
        is_af = labels[idx] == 1
        record_name = records[idx]  
        
        label_text = "AFIB (Atrial Fibrillation)" if is_af else "Normal Rhythm"
        color = '#e74c3c' if is_af else '#27ae60'
        
        plt.plot(time, signal, color=color, linewidth=1.2)
        
        is_flat = np.all(np.abs(signal) < 1e-5)
        title_extra = " [WARNING: FLAT SEGMENT]" if is_flat else ""
        
        plt.title(f"Record: {record_name} | Segment Index: {idx} | {label_text}{title_extra}", 
                  fontsize=13, fontweight='bold')
        plt.ylabel("Amplitude (mV)")
        plt.xlabel("Time (seconds)")
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()

def extractData(ecg_signal, ann_atr, fs, record_name, segment_length=2400):
    
    n_samples = len(ecg_signal)
    if n_samples < segment_length:
        return None, None, None, True

    ground_truth_samples = ann_atr.sample
 
    labels = [label.replace('(', '').replace(')', '') for label in ann_atr.aux_note]

    if not ground_truth_samples.any():
        return None, None, None, True

    all_segments = []
    all_labels = []
    all_record_names = [] 

    stats = {
        'total_raw': 0,
        'nan_flat_removed_total': 0,
        'nan_flat_removed_af': 0,
        'quality_removed_total': 0,
        'quality_removed_af': 0,
        'final_valid': 0
    }

    for i in range(len(ground_truth_samples)):
        start_sample = ground_truth_samples[i]
        end_sample = ground_truth_samples[i+1] if i+1 < len(ground_truth_samples) else n_samples
   
        current_label = labels[i]
        if 'AFIB' in current_label:
            label_value = 1  # Atrial Fibrillation
        elif 'N' in current_label:
            label_value = 0  # Normal/Other
        else:
            continue 

        region_signal = ecg_signal[start_sample:end_sample]
        num_segments = len(region_signal) // segment_length
        stats['total_raw'] += num_segments

        if num_segments > 0:
            for j in range(num_segments):
                start_seg = j * segment_length
                end_seg = (j + 1) * segment_length
                raw_segment = region_signal[start_seg:end_seg]
                
                if np.isnan(raw_segment).any() or np.std(raw_segment) < 1e-5:
                    stats['nan_flat_removed_total'] += 1
                    if label_value == 1:
                        stats['nan_flat_removed_af'] += 1
                    continue
                processed_segment = signal_preprocess_ecg.preprocessing_ECG(raw_segment)
                if mask_filter(processed_segment):
                    stats['quality_removed_total'] += 1
                    if label_value == 1:
                        stats['quality_removed_af'] += 1
                    continue

                stats['final_valid'] += 1
                all_segments.append(processed_segment.reshape(1, -1))
                all_labels.append(label_value)
                all_record_names.append(record_name) 

    if not all_segments:
        return None, None, None, True, stats

    final_segments = np.vstack(all_segments)
    final_labels = np.array(all_labels)
    final_record_names = np.array(all_record_names)

    print(f"--- Extraction Complete for {record_name} ---")
    print(f"Total segments: {final_segments.shape[0]}")
    
    return final_segments, final_labels, final_record_names, False, stats

    
def loadData(data_path="/home/linhhima/Pre_processing_data/Datasets/mit-bih-AF"):
    all_segments_list = []
    all_labels_list = []
    all_records_list = []

    record_files = sorted(list(set([f.split('.')[0] for f in os.listdir(data_path) if f.endswith('.dat')])))
    g_stats = {
        'raw': 0, 
        'nan_rem': 0, 'nan_af_rem': 0, 
        'sqi_rem': 0, 'sqi_af_rem': 0,
        'valid': 0
    }
    total_af_global = 0
    total_non_af_global = 0
    total_records = 0
    for record_name in record_files:
        hea_path = os.path.join(data_path, record_name + ".hea")
        atr_path = os.path.join(data_path, record_name + ".atr")
        
        if not os.path.exists(hea_path) or not os.path.exists(atr_path):
            print(f"Skipping {record_name}: Missing .hea or .atr file.")
            continue  
            
        print("====================================================")
        print(f"Processing record: {record_name}")

        try:
            
            record_data = wfdb.rdrecord(os.path.join(data_path, record_name), channels=[0])  
            ecg = record_data.p_signal[:, 0]
            Fs = record_data.fs # 250 Hz
            
            ann_atr = wfdb.rdann(os.path.join(data_path, record_name), 'atr')  

            segments, labels, names, error, rec_stats = extractData(
                ecg, ann_atr, Fs, record_name, segment_length=SLICE_LENGTH
            )

            if rec_stats:
                g_stats['raw'] += rec_stats['total_raw']
                g_stats['nan_rem'] += rec_stats['nan_flat_removed_total']
                g_stats['nan_af_rem'] += rec_stats['nan_flat_removed_af']
                g_stats['sqi_rem'] += rec_stats['quality_removed_total']
                g_stats['sqi_af_rem'] += rec_stats['quality_removed_af']
                g_stats['valid'] += rec_stats['final_valid']

            if not error and segments is not None:
                all_segments_list.append(segments)
                all_labels_list.append(labels)
                all_records_list.append(names)

                num_af = np.sum(labels == 1)
                num_non_af = np.sum(labels == 0)
                
                total_af_global += num_af
                total_non_af_global += num_non_af
                total_records += 1

                print(f"Record {record_name} Statistics:")
                print(f"  - AF segments: {num_af}")
                print(f"  - Non-AF segments: {num_non_af}")
            else:
                print(f"Record {record_name}: No valid segments extracted.")

        except Exception as e:
            print(f"Error processing {record_name}: {e}")
    print("all segments list: ", len(all_segments_list))
    if all_segments_list:
        final_segments = np.vstack(all_segments_list)
        final_labels = np.concatenate(all_labels_list)
        final_names = np.concatenate(all_records_list)
        print("\n" + "="*65)
        print("      DETAILED PREPROCESSING DROPOUT REPORT")
        print("="*65)
        print(f"1. Total initial segments extracted:         {g_stats['raw']}")
        
        print(f"\n2. Excluded due to NaN/Flat Line errors:")
        print(f"   - Total segments removed:                 {g_stats['nan_rem']}")
        print(f"   - Of which were AF-labeled:               {g_stats['nan_af_rem']}")
        
        print(f"\n3. Excluded due to Signal Quality (SQI < 0.3):")
        print(f"   - Total segments removed:                 {g_stats['sqi_rem']}")
        print(f"   - Of which were AF-labeled:               {g_stats['sqi_af_rem']}")
        
        print("\n" + "="*50)
        print("                GLOBAL STATISTICS")
        print("="*50)
        print(f"Total Records Processed:     {total_records}")
        print(f"Total AF Segments (1):       {total_af_global}")
        print(f"Total Non-AF Segments (0):   {total_non_af_global}")
        print(f"Grand Total Segments:        {len(final_labels)}")
        print(f"Imbalance Ratio (AF/Total):  {total_af_global/len(final_labels):.2%}")
        print("="*50)

        return final_segments, final_labels, final_names
    else:
        print("No data extracted from the entire database.")
        return None, None, None

        
def save_train_test_split(segments, labels, names, save_path="./processed_data/", seed=42):
   
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    unique_records = np.unique(names)
    np.random.seed(seed)
    np.random.shuffle(unique_records)

    split_idx = int(len(unique_records) * 0.8)
    train_records = unique_records[:split_idx]
    test_records = unique_records[split_idx:]

    print(f"--- Splitting Details (Seed: {seed}) ---")
    print(f"Train Records ({len(train_records)}): {train_records}")
    print(f"Test Records ({len(test_records)}): {test_records}")

    train_mask = np.isin(names, train_records)
    test_mask = np.isin(names, test_records)

    train_ecgs = segments[train_mask]
    train_labels = labels[train_mask]
    train_names = names[train_mask]

    test_ecgs = segments[test_mask]
    test_labels = labels[test_mask]
    test_names = names[test_mask]

    np.savez_compressed(os.path.join(save_path, "MIT_BIH_train_data.npz"), 
                        ecgs=train_ecgs, labels=train_labels, records=train_names)

    np.savez_compressed(os.path.join(save_path, "MIT_BIH_test_data.npz"), 
                        ecgs=test_ecgs, labels=test_labels, records=test_names)

    print("\n--- Save Completed ---")
    print(f"Train: {train_ecgs.shape[0]} segments | AF: {np.sum(train_labels==1)}")
    print(f"Test:  {test_ecgs.shape[0]} segments | AF: {np.sum(test_labels==1)}")
    print(f"Files saved at: {save_path}")

 
        
if __name__ == "__main__":
    segments, labels, names = loadData()
    # if segments is not None:
        # visualize_af_vs_normal_separate(segments, labels, names, fs=TARGET_FS)
        # save_train_test_split(segments, labels, names, seed=42)

