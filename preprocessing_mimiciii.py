import os
import numpy as np
import scipy.io as scio
from preprocessing import DataVisualizer, SignalProcessor
import random
from typing import Dict, Any, Tuple, List
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

        # Aligment PPG and ECG
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
           
            ppg_slice = ppg_preprocessed[start_idx:start_idx + SLICE_LENGTH]
            ecg_slice = ecg_preprocessed[start_idx:start_idx + SLICE_LENGTH]
          
           
            if np.isnan(ppg_slice).any() or np.isnan(ecg_slice).any():
                start_idx += OVERLAP
                continue

            record_ppgs.append(ppg_slice)
            record_ecgs.append(ecg_slice)
            record_names.append(record_name)
            start_idx += OVERLAP
    print("Tổng số segments PPG sau khi cắt: ", len(record_ppgs))
    return record_ppgs, record_ecgs, record_names

def split_segments_and_save_by_record(total_data: Dict[str, Any], save_prefix: str, ratios: Tuple[float, float]):
    """
    Chia 80% segment mỗi record vào train, 20% vào test và lưu theo định dạng save_dict.
    Giả định: total_data["records"] chứa tên record tương ứng với từng segment.
    """
    train_ratio, test_ratio = ratios
    train_indices = []
    test_indices = []

    # 1. Lấy danh sách các record duy nhất để duyệt qua từng người một
    unique_records = list(set(total_data["records"]))
    # Chuyển records sang numpy để tìm kiếm index nhanh hơn
    record_array = np.array(total_data["records"])

    print(f"--- Bắt đầu chia dữ liệu (Tỉ lệ {train_ratio}:{test_ratio}) ---")

    for rec_name in unique_records:
        # Tìm tất cả vị trí (index) của các segment thuộc record này
        indices = np.where(record_array == rec_name)[0]
        
        # Xáo trộn các index này nếu bạn muốn dữ liệu ngẫu nhiên hơn (không theo thời gian)
        # np.random.shuffle(indices) 

        num_seg = len(indices)
        split_point = int(num_seg * train_ratio)

        # Đảm bảo mỗi bộ có ít nhất 1 segment
        if split_point == 0 and num_seg > 0:
            split_point = 1
        elif split_point == num_seg and num_seg > 1:
            split_point = num_seg - 1

        # Chia index
        train_indices.extend(indices[:split_point])
        test_indices.extend(indices[split_point:])

    # 2. Tạo thư mục lưu nếu chưa có
    save_dir = "processed_data"
    os.makedirs(save_dir, exist_ok=True)

    # 3. Tiến hành tạo save_dict và lưu cho từng bộ (Train/Test)
    splits = [("train", train_indices), ("test", test_indices)]

    for split_name, current_indices in splits:
        save_path = os.path.join(save_dir, f"{save_prefix}_{split_name}.npz")
        
        # Tạo định dạng dict như bạn yêu cầu
        save_dict = {
            "ecgs": [total_data["ecgs"][i] for i in current_indices],
            "ppgs": [total_data["ppgs"][i] for i in current_indices],
            "records": [total_data["records"][i] for i in current_indices]
        }

        # Lưu vào file .npz
        # Lưu ý: Vì list có thể không đồng nhất, nên ép kiểu sang array nếu cần
        np.savez(save_path, **save_dict)
        print(f"→ Đã lưu {split_name.upper()}: {len(current_indices)} mẫu tại '{save_path}'")

    print("Hoàn tất xử lý!")

if __name__ == "__main__":
    set_seed(SEED)
    record_ppgs, record_ecgs, record_names = load_and_slice_all_signals()
    total_data = {
        "ppgs": record_ppgs,
        "ecgs": record_ecgs,
        "records": record_names,
    }
    split_segments_and_save_by_record(total_data, "mimic3_v1_no_align_mm", (0.8, 0.2))
