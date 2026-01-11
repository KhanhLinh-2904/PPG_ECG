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
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from scipy.signal import correlate
# --- CÁC HẰNG SỐ (CONSTANTS) ---
SEED = 44
PAUSE_TIME = 0.001
FS = 125  # Tần số lấy mẫu mặc định (Default Sampling Frequency)
WINDOW_SECONDS = 10
STEP_SIZE = 10 # Thay vì 10, nên dùng giá trị là bội số của fs để tránh lỗi hiển thị/tính toán
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
# --- LỚP XỬ LÝ TÍN HIỆU (SIGNAL PROCESSING CLASS) ---

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
        ppg_bpf = self._butter_lowpass(ppg_signal, cutoff=10, order=5)
        ppg_ac = self._dc_removal(time, ppg_bpf)
        ppg_remove_spikes =  self.remove_large_spikes_auto(ppg_ac, sigma_factor=1)
        ppg_normalized = self.normalize_signal(ppg_remove_spikes)
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
            print(f"Lỗi: Độ dài PPG ({len(ppg_segment)}) và ECG ({len(ecg_segment)}) không khớp!")
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
        
        # 1. Kiểm tra độ dài dữ liệu
        if total_samples < self.window_samples:
            print(f"Record {record_name} quá ngắn ({total_samples} samples). Bỏ qua.")
            return

        # 2. Setup biểu đồ
        plt.ion()
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        ax.set_title(f"Record: {record_name} (FS={self.fs}Hz)")
        x_axis = np.linspace(0, self.window_seconds, self.window_samples)

        # Khởi tạo 2 đường line rỗng
        # Lưu ý: Set alpha thấp hơn (0.7) để nhìn thấy nếu chúng chồng lên nhau
        line_ppg, = ax.plot(x_axis, np.zeros(self.window_samples), color="blue", linewidth=1.5, label="PPG", alpha=0.7)
        line_ecg, = ax.plot(x_axis, np.zeros(self.window_samples), color="red", linewidth=1.5, label="ECG", alpha=0.7)

        ax.set_ylabel("Normalized Amplitude")
        ax.set_xlabel("Time in Window (seconds)")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.6)

        print("Bắt đầu visualization... Nhấn Ctrl+C để dừng.")

        try:
            # 3. Vòng lặp trượt (Sliding Window)
            for start_idx in range(0, total_samples - self.window_samples, self.step_size):
                # Kiểm tra nếu cửa sổ plot bị tắt thì dừng luôn vòng lặp
                if not plt.fignum_exists(fig.number):
                    break
                
                end_idx = start_idx + self.window_samples
                
                # Cắt dữ liệu
                window_ppg = ppg_signal[start_idx:end_idx]
                window_ecg = ecg_signal[start_idx:end_idx]
                
                # --- XỬ LÝ AN TOÀN DỮ LIỆU ---
                # Thay thế NaN bằng 0 để tránh lỗi vẽ biểu đồ
                window_ppg = np.nan_to_num(window_ppg)
                window_ecg = np.nan_to_num(window_ecg)

                # Cập nhật dữ liệu cho đường line
                line_ppg.set_ydata(window_ppg)
                line_ecg.set_ydata(window_ecg)
                
                # --- KEY FIX: DYNAMIC RESCALING (TỰ ĐỘNG CO GIÃN) ---
                # Tìm min/max chung của cả 2 tín hiệu trong khung hình này
                current_min = min(np.min(window_ppg), np.min(window_ecg))
                current_max = max(np.max(window_ppg), np.max(window_ecg))
                
                # Thêm lề (margin) 10% để đỉnh sóng không chạm sát mép
                margin = (current_max - current_min) * 0.1 if (current_max != current_min) else 1.0
                
                ax.set_ylim(current_min - margin, current_max + margin)
                # ----------------------------------------------------

                # Cập nhật tiêu đề với thời gian thực
                curr_time_start = start_idx / self.fs
                curr_time_end = end_idx / self.fs
                ax.set_title(f"Record: {record_name} | Time: {curr_time_start:.2f}s - {curr_time_end:.2f}s")

                # Vẽ lại
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                
                if self.pause_time > 0:
                    time.sleep(self.pause_time)
                    
        except KeyboardInterrupt:
            print("\nĐã dừng bởi người dùng (KeyboardInterrupt).")
        except Exception as e:
            print(f"Lỗi xảy ra: {e}")
        finally:
            plt.close(fig)
            plt.ioff() # Tắt chế độ interactive
            print("Visualization hoàn tất.")

def get_max_cross_correlation_score(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0 
    x_norm = (x - np.mean(x)) / (np.std(x) * len(x))
    y_norm = (y - np.mean(y)) / np.std(y)
    corr = correlate(x_norm, y_norm, mode='full')
    return np.max(corr)

def compute_similarity_matrix(ecg_array):
    n_segments = ecg_array.shape[0]
    sim_matrix = np.zeros((n_segments, n_segments))
    
    for i in range(n_segments):
        for j in range(i, n_segments):
            if i == j:
                sim_matrix[i, j] = 1.0 
            else:
                score = get_max_cross_correlation_score(ecg_array[i], ecg_array[j])
                sim_matrix[i, j] = score
                sim_matrix[j, i] = score 
    return sim_matrix


def group_ecg_segment(record: Dict[str, List[Any]], threshold: float = 0.7) -> Dict[str, List[Any]]:
    results: Dict[str, List[Any]] = {
        "ppgs": [],
        "ecgs": [],
        "groupIDs": [],
        "labels": [],
        "records": []
    }

    ecg_segments = record.get("ecgs", [])
    ppg_segments = record.get("ppgs", [])
    record_names = record.get("records", [])
    input_labels = record.get("labels", [])

    if not ecg_segments:
        return results

    n_samples = len(ecg_segments)

    ecg_array = np.array(ecg_segments)
    
    sim_matrix = compute_similarity_matrix(ecg_array)
    
    np.fill_diagonal(sim_matrix, 0)

    # 2. Xây dựng đồ thị
    adjacency_matrix = sim_matrix >= threshold
    num_connections = np.sum(adjacency_matrix) / 2
    print(f"Tổng số cạnh (kết nối >= {threshold}): {int(num_connections)}")
    
    G = nx.from_numpy_array(adjacency_matrix)

    all_cliques = list(nx.find_cliques(G))
    all_cliques.sort(key=len, reverse=True)

    final_groups = []
    seen_nodes = set()

    for clique in all_cliques:
        unique_members = [node for node in clique if node not in seen_nodes]
        
        if unique_members:
            seen_nodes.update(unique_members)
         
            final_groups.append(unique_members)
            
            if len(unique_members) > 1:
                print(f" (Size {len(unique_members)}): {unique_members}")

    all_indices = set(range(n_samples))
    leftovers = list(all_indices - seen_nodes)
    
    if leftovers:
        for node in leftovers:
            final_groups.append([node])

    for group_id, group_indices in enumerate(final_groups):
        for idx in group_indices:
            results["groupIDs"].append(group_id)
            results["ecgs"].append(ecg_segments[idx])
            results["ppgs"].append(ppg_segments[idx])
            results["labels"].append(0)
            results["records"].append(record_names[idx])
    return results


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


        ppg_preprocessed = processor.align_signals_cross_correlation(ecg_preprocessed, ppg_preprocessed)[0]
        # visualizer.visualize_sliding_record(record_name, ppg_preprocessed, ecg_preprocessed)
        # visualizer.visualize_sliding_record(record_name, ppg_preprocessed, ppg)
        visualizer.visualize_specific_segment(record_name, ppg_preprocessed[:samples_to_plot], ecg_preprocessed[:samples_to_plot])



    #     min_len = min(len(ppg_preprocessed), len(ecg_preprocessed))
    #     print("Min length of signals: ", min_len)
    #     if min_len < SLICE_LENGTH:
    #         continue
    #     start_idx = 0
    #     prev_ppg = 0
    #     prev_ecg = 0
    #     while start_idx + SLICE_LENGTH <= min_len:
           
    #         ppg_slice = ppg_preprocessed[start_idx:start_idx + SLICE_LENGTH]
    #         ecg_slice = ecg_preprocessed[start_idx:start_idx + SLICE_LENGTH]
    #         if start_idx == 0:
    #             prev_ppg = ppg_slice
    #             prev_ecg = ecg_slice
    #         else:
    #             prev_ppg = record_ppgs[-1]
    #             prev_ecg = record_ecgs[-1]
    #         ppg_corr = get_max_cross_correlation_score(prev_ppg, ppg_slice)    
    #         ecg_corr = get_max_cross_correlation_score(prev_ecg, ecg_slice)  
    #         print("-----------------------")
    #         print("ppg_corr: ", ppg_corr)  
    #         print("ecg_corr: ", ecg_corr)  
    #         print("-----------------------")

    #         # or ppg_corr <= 0.7 or ecg_corr <= 0.3
    #         if np.isnan(ppg_slice).any() or np.isnan(ecg_slice).any():
    #             start_idx += OVERLAP
    #             continue

    #         record_ppgs.append(ppg_slice)
    #         record_ecgs.append(ecg_slice)
    #         record_names.append(record_name)
    #         start_idx += OVERLAP
    # print("Tổng số segments PPG sau khi cắt: ", len(record_ppgs))
    # return record_ppgs, record_ecgs, record_names


def split_segments_and_save(total_data: Dict[str, Any], save_prefix: str, ratios: Tuple[float, float]):
  
    os.makedirs("datasets", exist_ok=True)
    
    # 1. Gom nhóm
    # original_records = total_data.get("records")
    
    total_data = group_ecg_segment(total_data, threshold=0.9) 
    all_group_ids = np.array(total_data["groupIDs"])

    unique_groups = np.unique(all_group_ids)
    n_groups = len(unique_groups)
    
    print(f"\n--- THỐNG KÊ TỔNG QUÁT ---")
    print(f"Tổng số lượng Group ID: {n_groups}")

    # 3. Shuffle danh sách GROUP ID
    np.random.shuffle(unique_groups)
    
    # 4. Tính toán chia nhóm
    n_train_groups = int(ratios[0] * n_groups)
    
    train_group_ids = unique_groups[:n_train_groups]
    test_group_ids = unique_groups[n_train_groups:]

    # =================================================================
    # --- ĐOẠN CODE THỐNG KÊ MỚI BỔ SUNG ---
    # =================================================================
    
    # Bước A: Đếm kích thước của tất cả các nhóm trong dữ liệu gốc
    # unique_ids: danh sách ID, counts: số lượng phần tử của ID đó
    unique_ids_all, counts_all = np.unique(all_group_ids, return_counts=True)
    
    # Bước B: Xác định ID nào là Single (size=1) và ID nào là Cluster (size>1)
    single_ids = unique_ids_all[counts_all == 1]
    # cluster_ids = unique_ids_all[counts_all > 1] # Không cần dùng, nhưng để hiểu logic
    
    # Bước C: Đếm số lượng Single trong tập Train
    # np.intersect1d tìm các phần tử chung giữa 2 mảng (IDs của Train giao với IDs của Singles)
    n_singles_train = len(np.intersect1d(train_group_ids, single_ids))
    n_clusters_train = len(train_group_ids) - n_singles_train
    
    # Bước D: Đếm số lượng Single trong tập Test
    n_singles_test = len(np.intersect1d(test_group_ids, single_ids))
    n_clusters_test = len(test_group_ids) - n_singles_test
    
    print(f"\n--- CHI TIẾT PHÂN BỐ (Train/Test) ---")
    print(f"TRAIN Set ({len(train_group_ids)} nhóm):")
    print(f"  ✅ Clusters (>1 phần tử): {n_clusters_train} nhóm")
    print(f"  ⚠️ Singles  (1 phần tử):  {n_singles_train} nhóm")
    
    print(f"TEST Set ({len(test_group_ids)} nhóm):")
    print(f"  ✅ Clusters (>1 phần tử): {n_clusters_test} nhóm")
    print(f"  ⚠️ Singles  (1 phần tử):  {n_singles_test} nhóm")
    # =================================================================

    # 5. Map ngược từ Group ID về Segment Index
    train_indices = np.where(np.isin(all_group_ids, train_group_ids))[0].tolist()
    test_indices = np.where(np.isin(all_group_ids, test_group_ids))[0].tolist()
    
    # 6. Shuffle lại index trong từng tập
    random.shuffle(train_indices)
    random.shuffle(test_indices)

    print(f"\n--- KẾT QUẢ SỐ LƯỢNG SEGMENTS ---")
    print(f"Train: {len(train_indices)} segments")
    print(f"Test:  {len(test_indices)} segments")

    splits = {
        "train": train_indices,
        "test": test_indices
    }
    
    # 7. Lưu file
    for split_name, current_indices in splits.items():
        save_path = f"datasets/{save_prefix}_{split_name}.npz"
        
        if not current_indices:
            print(f"⚠️ Tập {split_name} rỗng!")
            continue

        save_dict = {
            "ecgs": [total_data["ecgs"][i] for i in current_indices],
            "ppgs": [total_data["ppgs"][i] for i in current_indices],
            "groupIDs": [total_data["groupIDs"][i] for i in current_indices],
            "labels": [total_data["labels"][i] for i in current_indices],
            "records": [total_data["records"][i] for i in current_indices]
        }

        np.savez(save_path, **save_dict)
        print(f"→ Đã lưu {split_name.upper()}: {len(current_indices)} mẫu tại '{save_path}'")

if __name__ == "__main__":
    set_seed(SEED)
    datapath = "/home/linhhima/Pre_processing_data/Datasets/mimic_perform_non_af_wfdb" 
    # datapath = "/home/linhhima/Pre_processing_data/Datasets/mimic_perform_af_wfdb"
    # all_records = get_all_records(datapath)
   
    # records_sample = ['mimic_perform_non_af_001', 'mimic_perform_non_af_002', 'mimic_perform_non_af_003', 
    #                   'mimic_perform_non_af_007', 'mimic_perform_non_af_009', 'mimic_perform_non_af_015',
    #                   'mimic_perform_non_af_005', 'mimic_perform_non_af_013', 'mimic_perform_non_af_016']
    # records_sample = ['mimic_perform_non_af_002']
    records_sample = ['mimic_perform_non_af_001', 'mimic_perform_non_af_002', 'mimic_perform_non_af_013', 'mimic_perform_non_af_016']
    # records_sample = ['mimic_perform_non_af_001', 'mimic_perform_non_af_002',
    #                 'mimic_perform_non_af_003', 'mimic_perform_non_af_004',
    #                 'mimic_perform_non_af_005', 'mimic_perform_non_af_006',
    #                 'mimic_perform_non_af_007', 'mimic_perform_non_af_008',
    #                 'mimic_perform_non_af_009', 'mimic_perform_non_af_010',
    #                 'mimic_perform_non_af_011', 'mimic_perform_non_af_012',
    #                 'mimic_perform_non_af_013', 'mimic_perform_non_af_014',
    #                 'mimic_perform_non_af_015', 'mimic_perform_non_af_016']
    
    # records_sample = ['mimic_perform_af_001', 'mimic_perform_af_002',
    #                 'mimic_perform_af_003', 'mimic_perform_af_004',
    #                 'mimic_perform_af_005', 'mimic_perform_af_006',
    #                 'mimic_perform_af_007', 'mimic_perform_af_008',
    #                 'mimic_perform_af_009', 'mimic_perform_af_010',
    #                 'mimic_perform_af_011', 'mimic_perform_af_012',
    #                 'mimic_perform_af_013', 'mimic_perform_af_014',
    #                 'mimic_perform_af_015', 'mimic_perform_af_016',
    #                 'mimic_perform_af_017', 'mimic_perform_af_018',
    #                 'mimic_perform_af_019'
    #                 ]
    # records_sample = ['mimic_perform_af_001', 'mimic_perform_af_002',
    #                 'mimic_perform_af_003', 'mimic_perform_af_004',
    #                 'mimic_perform_af_010',
    #                  'mimic_perform_af_012',
    #                 'mimic_perform_af_014',
    #                 'mimic_perform_af_015', 'mimic_perform_af_016',
    #                  'mimic_perform_af_018',
    #                 'mimic_perform_af_019'
    #                 ]
    non_af_data_ppg, non_af_data_ecg, record_names = load_and_slice_all_signals(datapath, records_sample)
    print("Tổng số segments PPG thu được: ", len(non_af_data_ppg))
    total_data = {
    "ppgs": non_af_data_ppg,
    "ecgs": non_af_data_ecg,
    "groupIDs": [],          # Sẽ được điền bởi group_ecg_segment
    "records": record_names, # Danh sách tên record tương ứng
    }   
    split_segments_and_save(total_data, save_prefix="normal", ratios=(0.8, 0.2))
