import numpy as np
from scipy import signal
from scipy.signal import medfilt
from scipy.interpolate import CubicSpline
from typing import List, Tuple, Dict, Any
import os
import wfdb
import matplotlib.pyplot as plt
import time
import random
from collections import defaultdict
from utils import calculate_bsqi, calculate_sq_mask
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
class SignalProcessor:
    """Chứa các hàm lọc, chuẩn hóa và tiền xử lý tín hiệu."""
    def __init__(self, fs: float):
        self.fs = fs

    def _butter_bandpass(self, data: np.ndarray, lowcut: float, highcut: float, order: int = 5) -> np.ndarray:
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq 
        low = np.clip(low, 0, 0.99)
        high = np.clip(high, 0, 0.99)
        
        b, a = signal.butter(order, [low, high], btype="band")
        y = signal.filtfilt(b, a, data)
        return y

    def _butter_lowpass(self, data: np.ndarray, cutoff: float=10, order: int = 5) -> np.ndarray:
        nyq = 0.5 * self.fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        y = signal.filtfilt(b, a, data)
        return y

    def _notch_filter_ecg(self, data: np.ndarray, notch_freq: float = 50.0, Q: float = 30.0) -> np.ndarray:
        nyq = 0.5 * self.fs
        w0 = notch_freq / nyq
        if w0 >= 1.0:
             print(f"⚠️ Notch filter {notch_freq}Hz is too high for FS={self.fs}Hz. Skipping.")
             return data
        b, a = signal.iirnotch(w0, Q)
        y = signal.filtfilt(b, a, data)
        return y

    def _get_prominence_threshold(self, signal_data: np.ndarray) -> float:
        temp = signal_data.copy()
        temp -= np.min(temp)
        return np.median(temp) * 0.5 

    
    def _dc_removal(self, time: np.ndarray, signal_data: np.ndarray) -> np.ndarray:
      
        inverted_signal = -1 * signal_data
        prominence_est = self._get_prominence_threshold(inverted_signal)
        distance = int(0.4 * self.fs) 
        feet_indices, _ = signal.find_peaks(inverted_signal, distance=distance, prominence=prominence_est)
        if len(feet_indices) < 2:
            return signal_data - np.mean(signal_data)

        anchored_indices = np.concatenate(([0], feet_indices, [len(signal_data)-1]))
        feet_values = signal_data[feet_indices]
        val_start = feet_values[0]
        val_end = feet_values[-1]
        anchored_values = np.concatenate(([val_start], feet_values, [val_end]))
      
        dc_spline = CubicSpline(time[anchored_indices], anchored_values)
        dc_component = dc_spline(time)
        ac_component = signal_data - dc_component
        
        return ac_component
 
    def remove_large_spikes_auto(self, signal, sigma_factor=5):
        smoothed_signal = medfilt(signal, kernel_size=3)
        diff = np.abs(signal - smoothed_signal)
        threshold = np.mean(diff) + sigma_factor * np.std(diff)
        spike_locations = diff > threshold
        cleaned_signal = np.where(spike_locations, smoothed_signal, signal)
        return cleaned_signal
    
    # def normalize_signal(self, signal_data: np.ndarray) -> np.ndarray:
    #     min_val = np.min(signal_data)
    #     max_val = np.max(signal_data)
    #     if max_val - min_val < 1e-8:
    #         return np.zeros_like(signal_data)
    #     return (signal_data - min_val) / (max_val - min_val)


    def normalize_signal(self, signal_data: np.ndarray) -> np.ndarray:
        mean_val = np.mean(signal_data)
        std_val = np.std(signal_data)
        if std_val < 1e-8:
            return np.zeros_like(signal_data)
        return (signal_data - mean_val) / std_val
    
    def align_signals_cross_correlation(self, ecg: np.ndarray, ppg: np.ndarray) -> Tuple[np.ndarray, int]:
        correlation = signal.correlate(ecg, ppg, mode="full")
        lags = signal.correlation_lags(len(ecg), len(ppg), mode="full")
        optimal_lag = lags[np.argmax(correlation)]
        aligned_ppg = np.roll(ppg, shift=optimal_lag)
        return aligned_ppg, optimal_lag 

    def preprocessing_PPG(self, ppg_signal: np.ndarray) -> np.ndarray:
        time = np.arange(len(ppg_signal)) / self.fs
        ppg_ac = self._dc_removal(time, ppg_signal)
        ppg_normalized = self.normalize_signal(ppg_ac)
        return ppg_normalized

    def preprocessing_ECG(self, ecg_signal: np.ndarray) -> np.ndarray:
        filtered_bandpass = self._butter_bandpass(ecg_signal, lowcut=0.5, highcut=100.0, order=5)
        filtered_notch = self._notch_filter_ecg(filtered_bandpass, notch_freq=50, Q=30)
        ecg_normalized = self.normalize_signal(filtered_notch)
        return ecg_normalized

# --- (DATA VISUALIZER CLASS) ---

class DataVisualizer:

    def __init__(self, fs: float, window_seconds: int, pause_time: float, step_size: int):
        self.fs = fs
        self.window_seconds = window_seconds
        self.pause_time = pause_time
        self.step_size = step_size
        self.window_samples = int(fs * window_seconds)

    def visualize_specific_segment(self, record_name: str, ppg_segment: np.ndarray, ecg_segment: np.ndarray):
      
        if len(ppg_segment) != len(ecg_segment):
            print(f"Error: Length of PPG ({len(ppg_segment)}) and ECG ({len(ecg_segment)}) do not match!")
            return
        
        ppg_clean = np.nan_to_num(ppg_segment)
        ecg_clean = np.nan_to_num(ecg_segment)
        
        num_samples = len(ppg_clean)
        duration = num_samples / self.fs
        x_axis = np.linspace(0, duration, num_samples)

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        ax.plot(x_axis, ppg_clean, color="blue", linewidth=1.5, label="PPG Segment", alpha=0.8)
        ax.plot(x_axis, ecg_clean, color="red", linewidth=1.5, label="ECG Segment", alpha=0.8)

        current_min = min(np.min(ppg_clean), np.min(ecg_clean))
        current_max = max(np.max(ppg_clean), np.max(ecg_clean))
        
        margin = (current_max - current_min) * 0.1 if (current_max != current_min) else 1.0
        ax.set_ylim(current_min - margin, current_max + margin)

        ax.set_title(f"Record: {record_name} | Length: {duration:.2f}s ({num_samples} samples)")
        ax.set_ylabel("Amplitude")
        ax.set_xlabel("Time (seconds)")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.show()

    def visualize_sliding_record(self, record_name: str, ppg_signal: np.ndarray, ecg_signal: np.ndarray):
        total_samples = len(ppg_signal)
        
        if total_samples < self.window_samples:
            print(f"Record {record_name} quá ngắn ({total_samples} samples). Bỏ qua.")
            return

        plt.ion()
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        ax.set_title(f"Record: {record_name} (FS={self.fs}Hz)")
        x_axis = np.linspace(0, self.window_seconds, self.window_samples)

        line_ppg, = ax.plot(x_axis, np.zeros(self.window_samples), color="blue", linewidth=1.5, label="PPG", alpha=0.7)
        line_ecg, = ax.plot(x_axis, np.zeros(self.window_samples), color="red", linewidth=1.5, label="ECG", alpha=0.7)

        ax.set_ylabel("Normalized Amplitude")
        ax.set_xlabel("Time in Window (seconds)")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.6)


        try:
            # 3. (Sliding Window)
            for start_idx in range(0, total_samples - self.window_samples, self.step_size):
                if not plt.fignum_exists(fig.number):
                    break
                
                end_idx = start_idx + self.window_samples
                
                window_ppg = ppg_signal[start_idx:end_idx]
                window_ecg = ecg_signal[start_idx:end_idx]
                
                window_ppg = np.nan_to_num(window_ppg)
                window_ecg = np.nan_to_num(window_ecg)

                line_ppg.set_ydata(window_ppg)
                line_ecg.set_ydata(window_ecg)
                
                current_min = min(np.min(window_ppg), np.min(window_ecg))
                current_max = max(np.max(window_ppg), np.max(window_ecg))
                
                margin = (current_max - current_min) * 0.1 if (current_max != current_min) else 1.0
                
                ax.set_ylim(current_min - margin, current_max + margin)

                curr_time_start = start_idx / self.fs
                curr_time_end = end_idx / self.fs
                ax.set_title(f"Record: {record_name} | Time: {curr_time_start:.2f}s - {curr_time_end:.2f}s")

                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                
                if self.pause_time > 0:
                    time.sleep(self.pause_time)
                    
        except KeyboardInterrupt:
            print("\nStopped by user (KeyboardInterrupt).")
        except Exception as e:
            print(f"Error occurred: {e}")
        finally:
            plt.close(fig)
            plt.ioff() 
            print("Visualization complete.")

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
           
            ppg_slice = ppg_preprocessed[start_idx:start_idx + SLICE_LENGTH]
            ecg_slice = ecg_preprocessed[start_idx:start_idx + SLICE_LENGTH]
             
           
            if np.isnan(ppg_slice).any() or np.isnan(ecg_slice).any():
                start_idx += OVERLAP
                print("Found NaN values, skipping this segment.")
                continue
            if mask_filter(ecg_slice, ppg_slice):
                start_idx += OVERLAP
                print("Segment failed quality check, skipping.")
                continue
            record_ppgs.append(ppg_slice)
            record_ecgs.append(ecg_slice)
            record_names.append(record_name)
            start_idx += OVERLAP
        print("Number of segments extracted from this record: ", len(record_ppgs))
    print("Total number of PPG segments after slicing: ", len(record_ppgs))
    return record_ppgs, record_ecgs, record_names

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
    non_af_data_ppg, non_af_data_ecg, non_af_records = load_and_slice_all_signals(datapath_non_af, records_sample_non_af)
    af_data_ppg, af_data_ecg, af_records = load_and_slice_all_signals(datapath_af, records_sample_af)

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
    save_data(total_data, "total_z")
    # split_segments_and_save(total_data, save_prefix="normal_remove24", ratios=(0.8, 0.2))
    # split_segments_and_save_by_record(total_data, save_prefix="record", ratios=(0.8, 0.2))
    # split_segments_and_save(total_data, save_prefix="total_z", ratios=(0.8, 0.2))
    # split_segments_and_save_by_record(total_data, save_prefix="total_record_mm", ratios=(0.8, 0.2))

