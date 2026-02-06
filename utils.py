import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
# Cài đặt nếu chưa có: pip install biosppy
from biosppy.signals import ecg 

def detect_ecg_features(signal, sampling_rate=250):
    rpeaks_indices = ecg.hamilton_segmenter(signal=signal, sampling_rate=sampling_rate)['rpeaks']
   
    all_peaks, _ = find_peaks(signal, distance=sampling_rate*0.2) # Khoảng cách tối thiểu giữa các đỉnh ~200ms
    
    all_valleys, _ = find_peaks(-signal, distance=sampling_rate*0.2)

    opeaks_indices = np.setdiff1d(np.union1d(all_peaks, all_valleys), rpeaks_indices)
    
    return rpeaks_indices, opeaks_indices

if __name__ == "__main__":
    # --- Demo với dữ liệu giả lập ---
    fs = 250  # Tần số lấy mẫu
    t = np.linspace(0, 2, 2 * fs)
    # Tạo tín hiệu mô phỏng đơn giản (có thể thay bằng real_B trong code của bạn)
    dummy_ecg = np.sin(2 * np.pi * 1.5 * t) + 0.5 * np.sin(2 * np.pi * 15 * t) 

    r_idx, o_idx = detect_ecg_features(dummy_ecg, sampling_rate=fs)

    # Trực quan hóa
    plt.figure(figsize=(12, 4))
    plt.plot(dummy_ecg, label='ECG Signal', color='black', alpha=0.5)
    plt.scatter(r_idx, dummy_ecg[r_idx], color='red', label='R-peaks (Hamilton)', zorder=5)
    plt.scatter(o_idx, dummy_ecg[o_idx], color='blue', marker='x', label='Other peaks/valleys (Scipy)')
    plt.legend()
    plt.title("ECG Feature Detection")
    plt.show()