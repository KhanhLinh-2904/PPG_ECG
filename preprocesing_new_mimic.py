import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from utils import calculate_bsqi, calculate_sq_mask
def mask_filter(ecg, ppg): 
    sq_mask_array = calculate_sq_mask(ppg, fs=125)
    ppg_sqi = np.mean(sq_mask_array)
    ecg_sqi = calculate_bsqi(ecg, fs=125)
    if ppg_sqi < 0.3 or ecg_sqi < 0.3:
        print("ppg_sqi: ", ppg_sqi)
        print("ecg_sqi: ", ecg_sqi)
        return True
    return False
def read_mat_scipy(path):
    # squeeze_me giúp loại bỏ các chiều dư thừa (1,1,N) -> (N,)
    mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    
    # Lấy cell array
    raw_data = mat['all_clean_data']
    
    # Nếu chỉ có 1 segment, raw_data có thể không phải mảng, ta ép kiểu
    if not isinstance(raw_data, np.ndarray):
        raw_data = np.array([raw_data])
    
    clean_list = []
    for seg in raw_data:
        d = {
            'record_name': str(seg.record_name),
            'segment_index': int(seg.segment_index),
            'fs': float(seg.fs),
            'sqi_ecg': float(seg.sqi_ecg),
            'sqi_ppg': float(seg.sqi_ppg),
            'ecg': seg.ecg,
            'ppg': seg.ppg
        }
        clean_list.append(d)
    return clean_list

def plot_signals(clean_list, num_segments=1, duration_sec=10):
    """
    Hiển thị cặp tín hiệu ECG và PPG.
    - num_segments: Số lượng segment muốn hiển thị (mặc định là 1).
    - duration_sec: Số giây muốn xem (mặc định 10 giây để nhìn rõ sóng).
    """
    if not clean_list:
        print("Danh sách dữ liệu trống!")
        return

    # Giới hạn số lượng vẽ nếu yêu cầu vượt quá dữ liệu hiện có
    num_to_draw = min(len(clean_list), num_segments)

    for i in range(num_to_draw):
        seg = clean_list[i]
        fs = seg['fs']
        ecg = seg['ecg']
        ppg = seg['ppg']
        print(f"Record: {seg['record_name']} | Segment: {seg['segment_index']}\n")
        mask_filter(ecg, ppg)
        # Tạo trục thời gian
        num_samples = int(min(len(ecg), duration_sec * fs))
        time = np.arange(num_samples) / fs

        # Khởi tạo đồ thị
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
        
        # Tiêu đề tổng quát
        fig.suptitle(f"Record: {seg['record_name']} | Segment: {seg['segment_index']}\n"
                     f"ECG SQI: {seg['sqi_ecg']:.2f} | PPG SQI: {seg['sqi_ppg']:.2f}", 
                     fontsize=14, fontweight='bold')

        # Vẽ ECG
        ax1.plot(time, ecg[:num_samples], color='#1f77b4', linewidth=1)
        ax1.set_ylabel('ECG (mV)', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_title('Electrocardiogram (ECG)', loc='left', color='blue')

        # Vẽ PPG
        ax2.plot(time, ppg[:num_samples], color='#d62728', linewidth=1)
        ax2.set_ylabel('PPG (Unit)', fontsize=12)
        ax2.set_xlabel('Time (seconds)', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.set_title('Photoplethysmogram (PPG)', loc='left', color='red')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

# --- CHẠY CHƯƠNG TRÌNH ---
file_path = 'record_clean.mat'
try:
    # 1. Nạp dữ liệu
    data = read_mat_scipy(file_path)
    print(f"✅ Đã nạp {len(data)} segments.")

    # 2. Hiển thị 3 đoạn đầu tiên, mỗi đoạn xem 15 giây
    plot_signals(data, num_segments=7, duration_sec=19.2)

except FileNotFoundError:
    print(f"❌ Không tìm thấy file: {file_path}")
except Exception as e:
    print(f"❌ Có lỗi xảy ra: {e}")