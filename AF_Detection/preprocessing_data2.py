import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
import tkinter as tk
import random
import torch

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Tắt cửa sổ chính của Tkinter
root = tk.Tk()
root.withdraw()

# =====================================================================
# THUẬT TOÁN PAN-TOMPKINS (Tìm đỉnh R)
# =====================================================================
def pan_tompkins_qrs(ecg_signal: np.ndarray, fs: int = 125):
    ecg_signal = np.array(ecg_signal).flatten()
    nyq = 0.5 * fs
    b, a = butter(1, [5.0 / nyq, 15.0 / nyq], btype='band')
    filtered_ecg = filtfilt(b, a, ecg_signal)
    
    diff_ecg = np.diff(filtered_ecg)
    diff_ecg = np.insert(diff_ecg, 0, diff_ecg[0])
    
    squared_ecg = diff_ecg ** 2
    
    window_width = int(0.15 * fs)
    integrated_ecg = np.convolve(squared_ecg, np.ones(window_width) / window_width, mode='same')
    
    threshold = np.mean(integrated_ecg)
    min_distance = int(0.3 * fs)
    peaks_integrated, _ = find_peaks(integrated_ecg, height=threshold, distance=min_distance)
    
    r_peaks = []
    search_window = int(0.05 * fs)
    for p in peaks_integrated:
        start = max(0, p - search_window)
        end = min(len(ecg_signal), p + search_window)
        if start < end:
            local_max = np.argmax(ecg_signal[start:end])
            r_peaks.append(start + local_max)
            
    return np.array(r_peaks)

def detect_ECG_beats_with_indices(ecg_signal):
    fs = 125
    r_indices = pan_tompkins_qrs(ecg_signal, fs=fs)
    if len(r_indices) < 2:
        return np.array([]), np.array([])
    rr_intervals = np.diff(r_indices) / fs
    return rr_intervals, r_indices

# =====================================================================
# HÀM PHÁT HIỆN RANH GIỚI SÓNG P (ĐÃ FIX LỖI WINDOW & FILTER)
# =====================================================================
def detect_p_wave_boundaries(signal, r_indices, fs=125):
    """
    Dò tìm Onset, Peak, Endset dựa trên Bandpass filter và cửa sổ lùi sâu.
    """
    p_data = []
    
    # [FIX 1] Mở rộng cửa sổ: Lùi 450ms để bắt được sóng P ở nhịp tim chậm
    search_start = int(0.45 * fs) 
    search_end = int(0.05 * fs)   
    
    # [FIX 2] Dùng Bandpass 0.5 - 10 Hz để ép đường nền phẳng tuyệt đối
    nyq = 0.5 * fs
    b, a = butter(2, [0.5 / nyq, 10.0 / nyq], btype='band')
    clean_signal = filtfilt(b, a, signal)

    for r in r_indices:
        start_idx = r - search_start
        end_idx = r - search_end
        
        if start_idx >= 0 and end_idx < len(signal):
            # Cắt cửa sổ từ tín hiệu đã lọc Bandpass
            search_window = clean_signal[start_idx:end_idx]
            detrended_window = search_window - np.mean(search_window)
            
            # Tìm các đỉnh
            peaks, _ = find_peaks(detrended_window, prominence=0.005)
            
            if len(peaks) > 0:
                peak_idx_local = peaks[np.argmax(detrended_window[peaks])]
                peak_val = detrended_window[peak_idx_local]
                
                # Cắt ranh giới ở mức 15% biên độ đỉnh
                threshold = 0.15 * peak_val
                
                onset_idx_local = 0
                for i in range(peak_idx_local, -1, -1):
                    if detrended_window[i] <= threshold:
                        onset_idx_local = i
                        break
                        
                endset_idx_local = len(detrended_window) - 1
                for i in range(peak_idx_local, len(detrended_window)):
                    if detrended_window[i] <= threshold:
                        endset_idx_local = i
                        break
                        
                abs_onset = start_idx + onset_idx_local
                abs_peak = start_idx + peak_idx_local
                abs_endset = start_idx + endset_idx_local
                
                # Để hiển thị hình dáng tự nhiên nhất, trích xuất lại từ tín hiệu GỐC
                original_waveform = signal[abs_onset:abs_endset+1]
                detrended_original = original_waveform - np.mean(original_waveform)
                
                p_data.append({
                    'onset': abs_onset,
                    'peak': abs_peak,
                    'endset': abs_endset,
                    'waveform': detrended_original
                })

    return p_data

# =====================================================================
# HÀM HIỂN THỊ BẢNG ĐIỀU KHIỂN (DASHBOARD)
# =====================================================================
def visualize_comprehensive_p_waves(signal, r_indices, label, record_index, fs=125):
    p_data = detect_p_wave_boundaries(signal, r_indices, fs)
    
    if len(p_data) == 0:
        print(f"Record {record_index}: Không tìm thấy sóng P rõ ràng.")
        return

    p_onsets = np.array([p['onset'] for p in p_data])
    p_peaks = np.array([p['peak'] for p in p_data])
    p_endsets = np.array([p['endset'] for p in p_data])
    
    fig, axs = plt.subplots(2, 1, figsize=(14, 10))
    time_ax = np.arange(len(signal)) / fs
    label_str = "AF" if label == 1 else "Non-AF"
    
    # ---------------------------------------------------------
    # ĐỒ THỊ 1: TÍN HIỆU GỐC & CÁC ĐIỂM ĐÁNH DẤU
    # ---------------------------------------------------------
    axs[0].plot(time_ax, signal, color='black', linewidth=1, alpha=0.8)
    axs[0].scatter(r_indices/fs, signal[r_indices], color='red', marker='v', s=60, zorder=3, label='Đỉnh R')
    
    axs[0].scatter(p_peaks/fs, signal[p_peaks], color='limegreen', marker='o', s=40, zorder=4, label='P-peak')
    axs[0].scatter(p_onsets/fs, signal[p_onsets], color='blue', marker='>', s=40, zorder=4, label='P-onset')
    axs[0].scatter(p_endsets/fs, signal[p_endsets], color='magenta', marker='<', s=40, zorder=4, label='P-endset')

    # Vẽ Highlight cho vùng diện tích thực sự của sóng P
    for i, (on, off) in enumerate(zip(p_onsets, p_endsets)):
        lbl = 'Ranh giới sóng P' if i == 0 else ""
        axs[0].axvspan(on/fs, off/fs, color='gold', alpha=0.3, label=lbl)

    axs[0].set_title(f"Record {record_index} [{label_str}] - Tín hiệu ECG & Định vị Sóng P (Đã Fix Cửa sổ)", fontweight='bold')
    axs[0].set_ylabel("Amplitude")
    axs[0].set_xlabel("Thời gian (giây)")
    axs[0].legend(loc="upper right")
    axs[0].grid(True, alpha=0.3)

    # ---------------------------------------------------------
    # ĐỒ THỊ 2: TRẢI PHẲNG CÁC SÓNG P ĐỂ ĐÁNH GIÁ MORPHOLOGY
    # ---------------------------------------------------------
    axs[1].set_title(f"Trải phẳng và Căn giữa {len(p_data)} Sóng P đã trích xuất", fontweight='bold', color='darkblue')
    axs[1].set_ylabel("Amplitude (Detrended)")
    axs[1].set_xlabel("Thời gian tính từ P-onset (giây)")
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(p_data)))
    
    for i, p_info in enumerate(p_data):
        waveform = p_info['waveform']
        t_wave = np.arange(len(waveform)) / fs
        axs[1].plot(t_wave, waveform, color=colors[i], alpha=0.7, linewidth=2)

    axs[1].axhline(0, color='black', linestyle='--', alpha=0.5)
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show(block=True)

if __name__ == "__main__":
    set_seed(42)
    
    dataset_path = "/home/linhhima/PPG_ECG/datasets/z_score_norm/total_mimic_af.npz"
    print(f"[*] Đang load dữ liệu từ {dataset_path} để kiểm tra Sóng P...")
    
    if os.path.exists(dataset_path):
        demo_data = np.load(dataset_path, allow_pickle=True)
        demo_ecgs = demo_data["ecgs"]
        demo_labels = demo_data["labels"]
        
        for i in range(len(demo_ecgs)):
            sig = demo_ecgs[i]
            lbl = demo_labels[i]
            
            # Chỉ hiển thị nhãn 0 (Non-AF) để quan sát sóng P
            if lbl == 1:  
                _, r_idx = detect_ECG_beats_with_indices(sig)
                
                if len(r_idx) >= 1:
                    visualize_comprehensive_p_waves(sig, r_idx, lbl, record_index=i, fs=125)
            
    else:
        print(f"[!] Không tìm thấy file {dataset_path}.")