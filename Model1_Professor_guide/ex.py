import torch
import numpy as np
import matplotlib
matplotlib.use('Agg') # Rất quan trọng khi chạy trên server
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from load_data import LoadData
import os

# --- CÁC CONFIG CỦA BẠN ---
TEST_DATA_PATH = '/home/linhhima/PPG_ECG/datasets/z_score_norm/total_mimic_af.npz'

def visualize_pan_tompkins(ecg_signal: np.ndarray, fs: int = 125, save_path="pan_tompkins_steps.png"):
    """
    Hàm mô phỏng và vẽ đồ thị 5 bước của thuật toán Pan-Tompkins.
    """
    ecg_signal = np.array(ecg_signal).flatten()
    
    # 1. Bandpass Filter (5 - 15 Hz)
    nyq = 0.5 * fs
    b, a = butter(1, [5.0 / nyq, 15.0 / nyq], btype='band')
    filtered_ecg = filtfilt(b, a, ecg_signal)
    
    # 2. Derivative (Đạo hàm)
    diff_ecg = np.diff(filtered_ecg)
    diff_ecg = np.insert(diff_ecg, 0, diff_ecg[0])
    
    # 3. Squaring (Bình phương)
    squared_ecg = diff_ecg ** 2
    
    # 4. Moving Window Integration (Tích phân cửa sổ trượt)
    window_width = int(0.15 * fs)
    integrated_ecg = np.convolve(squared_ecg, np.ones(window_width) / window_width, mode='same')
    
    # 5. Thresholding & Find Peaks trên mảng Tích phân
    threshold = np.mean(integrated_ecg)
    min_distance = int(0.3 * fs)
    peaks_integrated, _ = find_peaks(integrated_ecg, height=threshold, distance=min_distance)
    
    # 6. Back-search (Tìm ngược đỉnh R trên tín hiệu gốc)
    r_peaks = []
    search_window = int(0.05 * fs)
    for p in peaks_integrated:
        start = max(0, p - search_window)
        end = min(len(ecg_signal), p + search_window)
        if start < end:
            local_max = np.argmax(ecg_signal[start:end])
            r_peaks.append(start + local_max)
    r_peaks = np.array(r_peaks)

    # ==========================================
    # QUÁ TRÌNH VẼ ĐỒ THỊ (VISUALIZATION)
    # ==========================================
    # Tạo Time vector (trục X) theo giây
    time_ax = np.arange(len(ecg_signal)) / fs 
    
    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
    fig.suptitle('Pan-Tompkins Algorithm Step-by-Step Visualization', fontsize=18, fontweight='bold')

    # Bước 1: Tín hiệu gốc + Đỉnh R (Kết quả cuối cùng)
    axes[0].plot(time_ax, ecg_signal, color='black', label='Original ECG')
    axes[0].plot(time_ax[r_peaks], ecg_signal[r_peaks], 'ro', markersize=8, label='Detected R-peaks')
    axes[0].set_title('Step 0 & 6: Original Signal with Final Detected R-Peaks')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.5)

    # Bước 2: Bandpass Filter
    axes[1].plot(time_ax, filtered_ecg, color='blue')
    axes[1].set_title('Step 1: Bandpass Filtered (5-15 Hz) - Removes baseline wander & high-freq noise')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.5)

    # Bước 3: Derivative
    axes[2].plot(time_ax, diff_ecg, color='green')
    axes[2].set_title('Step 2: Derivative - Emphasizes high slopes (QRS complex)')
    axes[2].set_ylabel('Amplitude')
    axes[2].grid(True, alpha=0.5)

    # Bước 4: Squaring
    axes[3].plot(time_ax, squared_ecg, color='orange')
    axes[3].set_title('Step 3: Squaring - Makes all values positive & amplifies higher frequencies')
    axes[3].set_ylabel('Amplitude^2')
    axes[3].grid(True, alpha=0.5)

    # Bước 5: Integration + Threshold
    axes[4].plot(time_ax, integrated_ecg, color='purple', label='Integrated Signal')
    axes[4].axhline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.4f})')
    axes[4].plot(time_ax[peaks_integrated], integrated_ecg[peaks_integrated], 'x', color='black', markersize=10, label='Integrated Peaks')
    axes[4].set_title('Step 4 & 5: Moving Window Integration + Thresholding')
    axes[4].set_xlabel('Time (seconds)')
    axes[4].set_ylabel('Amplitude')
    axes[4].legend(loc='upper right')
    axes[4].grid(True, alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[*] Đã lưu ảnh visualization thành công tại: {save_path}")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Load dataset
    print("[*] Đang load dữ liệu từ dataset...")
    dataset = LoadData(TEST_DATA_PATH)
    
    # 2. Lấy 1 mẫu (sample) đầu tiên ra để visualize
    # Cấu trúc của LoadData thường trả về (ecg, ppg, label/record_name)
    sample_ecg, _, _ = dataset[300] 
    
    # Chuyển Tensor sang Numpy array nếu cần
    if isinstance(sample_ecg, torch.Tensor):
        sample_ecg = sample_ecg.numpy()
        
    # 3. CẮT NGẮN TÍN HIỆU ĐỂ DỄ NHÌN HƠN (Rất quan trọng!)
    # Nếu tín hiệu dài 2400 (19.2 giây), đồ thị sẽ bị nhiễu mắt. Ta chỉ lấy 800 điểm đầu (~6.4 giây)
    display_length = min(800, len(sample_ecg))
    sample_ecg_short = sample_ecg[:display_length]
    
    # 4. Gọi hàm vẽ đồ thị
    print(f"[*] Đang xử lý và vẽ đồ thị thuật toán Pan-Tompkins cho {display_length} điểm dữ liệu...")
    visualize_pan_tompkins(sample_ecg_short, fs=125, save_path="pan_tompkins_demo.png")