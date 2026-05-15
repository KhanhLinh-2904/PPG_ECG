import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
from scipy.fftpack import fft, fftfreq
import random
import torch
import tkinter as tk

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
# 1. TIỀN XỬ LÝ & TÌM ĐỈNH R (Chỉ dùng làm mốc để khoanh vùng sóng P)
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

# =====================================================================
# 2. PHÂN TÍCH VÙNG SÓNG P & LƯU TỌA ĐỘ ĐỈNH P (CHỈ TÍNH ENERGY NẾU CÓ ĐỈNH P)
# =====================================================================
def analyze_p_wave_for_af(signal, fs=125):
    r_indices = pan_tompkins_qrs(signal, fs)
    
    if len(r_indices) < 2:
        return 0.0, 0.0, [], []  
        
    nyq = 0.5 * fs
    b, a = butter(2, [0.5 / nyq, 15.0 / nyq], btype='band')
    clean_signal = filtfilt(b, a, signal)

    num_p_found = 0
    p_peaks_indices = []
    total_power_1_3 = 0.0
    total_power_3_10 = 0.0
    
    # Quét từ nhịp thứ 2 trở đi để biết khoảng cách R-R
    for i in range(1, len(r_indices)):
        r = r_indices[i]
        r_prev = r_indices[i-1] 
        
        rr_distance = r - r_prev
        
        # Áp dụng công thức cửa sổ hoàn toàn linh hoạt (Dynamic Window)
        search_start = int(rr_distance * 0.5)
        search_end = int(0.05 * rr_distance)
        
        start_idx = r - search_start
        end_idx = r - search_end
        
        if start_idx >= r_prev and end_idx < len(signal):
            p_window = clean_signal[start_idx:end_idx]
            p_detrended = p_window - np.mean(p_window)
            
            # --- TÌM SÓNG P ---
            peaks, _ = find_peaks(p_detrended, prominence=0.5)
            
            # --- FIX: CHỈ TÍNH NĂNG LƯỢNG KHI CÓ SÓNG P ĐƯỢC TÌM THẤY ---
            if len(peaks) > 0:
                num_p_found += 1
                
                # Lưu tọa độ P-peak
                peak_idx_local = peaks[np.argmax(p_detrended[peaks])]
                abs_p_peak = start_idx + peak_idx_local
                p_peaks_indices.append(abs_p_peak)
                
                # --- FFT NĂNG LƯỢNG (Dịch vào trong khối IF) ---
                windowed = p_detrended * np.hamming(len(p_detrended))
                N = len(windowed)
                freqs = fftfreq(N, 1/fs)[:N//2]
                energy = np.abs(fft(windowed))[:N//2] ** 2
                
                power_1_3Hz = np.sum(energy[(freqs >= 1.0) & (freqs < 3.0)])
                power_3_10Hz = np.sum(energy[(freqs >= 3.0)])
                
                energy_ratio = (power_3_10Hz / power_1_3Hz) if power_1_3Hz > 0 else 0.0
                total_power_1_3 += power_1_3Hz
                total_power_3_10 += power_3_10Hz

    # Số lượng nhịp thực tế được đưa vào phân tích (trừ đi nhịp đầu tiên)
    valid_beats = len(r_indices) - 1
    p_ratio = num_p_found / valid_beats if valid_beats > 0 else 0
    
    # Tính tỷ lệ năng lượng (Chỉ dựa trên các nhịp có sóng P)
    if total_power_1_3 == 0: 
        total_power_1_3 = 1e-6 
        
    avg_energy_ratio = (total_power_3_10 / total_power_1_3) if num_p_found > 0 else 0.0
    
    return p_ratio, avg_energy_ratio, p_peaks_indices, r_indices

# =====================================================================
# 3. CHẨN ĐOÁN & KIỂM THỬ
# =====================================================================
def run_pure_p_wave_classification(dataset_path, num_plots_to_show=5, filter_view="all"):
    """
    filter_view: 
        - "all"    : Xem tất cả các bản ghi
        - "af"     : Chỉ xem các bản ghi có thực tế là Rung nhĩ (AF)
        - "non_af" : Chỉ xem các bản ghi có thực tế là Bình thường (Non-AF)
    """
    print(f"[*] Đang load dữ liệu từ {dataset_path}...")
    if not os.path.exists(dataset_path):
        print("[!] Không tìm thấy file.")
        return

    data = np.load(dataset_path, allow_pickle=True)
    ecgs = data["ecgs"]
    labels = data["labels"]
    
    correct_af, correct_non_af = 0, 0
    total_af, total_non_af = 0, 0
    
    P_RATIO_THRESH = 0.7
    ENERGY_THRESH = 1.3

    print("\n[*] Đang tiến hành phân tích Sóng P...")
    
    plots_shown = 0 

    for i in range(len(ecgs)):
        sig = ecgs[i]
        true_lbl = labels[i]

        p_ratio, energy_ratio, p_peaks_idx, r_idx = analyze_p_wave_for_af(sig, fs=125)
        
        if p_ratio > P_RATIO_THRESH:
            pred_lbl = 0 # Non-AF
        else:
            pred_lbl = 1 # AF
            
        # Thống kê
        if true_lbl == 1:
            total_af += 1
            if pred_lbl == 1: correct_af += 1
        else:
            total_non_af += 1
            if pred_lbl == 0: correct_non_af += 1

        # =====================================================================
        # KHỐI LỆNH MỚI: VẼ ĐỒ THỊ HIỂN THỊ ĐỈNH P VÀ R
        # =====================================================================
        show_this_plot = False
        if filter_view == "all":
            show_this_plot = True
        elif filter_view == "af" and true_lbl == 1:
            show_this_plot = True
        elif filter_view == "non_af" and true_lbl == 0:
            show_this_plot = True

        if show_this_plot and plots_shown < num_plots_to_show and len(r_idx) > 0:
            fig, ax = plt.subplots(1, 1, figsize=(12, 4))
            time_ax = np.arange(len(sig)) / 125
            
            title_color = 'darkgreen' if true_lbl == pred_lbl else 'darkred'
            true_str = "AF" if true_lbl == 1 else "Non-AF"
            pred_str = "AF" if pred_lbl == 1 else "Non-AF"
            
            ax.plot(time_ax, sig, color='black', linewidth=1, alpha=0.8, label='ECG Signal')
            ax.scatter(r_idx/125, sig[r_idx], color='red', marker='v', s=60, zorder=3, label='Đỉnh R')
            
            if len(p_peaks_idx) > 0:
                p_peaks_arr = np.array(p_peaks_idx)
                ax.scatter(p_peaks_arr/125, sig[p_peaks_arr], color='limegreen', marker='o', s=50, zorder=4, label='Đỉnh P')

            for j in range(1, len(r_idx)):
                r_curr = r_idx[j]
                r_prev = r_idx[j-1]
                rr_dist = r_curr - r_prev
                
                s_time = max(0, (r_curr - int(rr_dist * 0.45)) / 125)
                e_time = max(0, (r_curr - int(0.05 * rr_dist)) / 125)
                
                ax.axvspan(s_time, e_time, color='gray', alpha=0.15)

            ax.set_title(f"Record {i} | Thực tế: {true_str} | Dự đoán: {pred_str}\nP-Ratio: {p_ratio:.2f}, Energy Ratio: {energy_ratio:.2f}", fontweight='bold', color=title_color)
            ax.set_xlabel("Thời gian (giây)")
            ax.set_ylabel("Amplitude")
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show(block=True)
            
            plots_shown += 1

    # =====================================================================
    # IN BẢNG TỔNG KẾT
    # =====================================================================
    print("\n" + "="*60)
    print(" 📊 TỔNG KẾT CHẨN ĐOÁN RUNG NHĨ (CHỈ DÙNG P-WAVE)")
    print("="*60)
    
    acc_af = (correct_af / total_af * 100) if total_af > 0 else 0
    acc_non_af = (correct_non_af / total_non_af * 100) if total_non_af > 0 else 0
    total_records = total_af + total_non_af
    overall_acc = ((correct_af + correct_non_af) / total_records * 100) if total_records > 0 else 0

    print(f"🔹 Tổng số bản ghi test : {total_records} records")
    print("-" * 60)
    print(f"🔴 RUNG NHĨ (AF):")
    print(f"   - Số ca thực tế         : {total_af} ca")
    print(f"   - Máy dự đoán ĐÚNG      : {correct_af} ca")
    print(f"   => Độ chính xác (Sens)  : {acc_af:.2f}%")
    print("-" * 60)
    print(f"🟢 BÌNH THƯỜNG (Non-AF):")
    print(f"   - Số ca thực tế         : {total_non_af} ca")
    print(f"   - Máy dự đoán ĐÚNG      : {correct_non_af} ca")
    print(f"   => Độ chính xác (Spec)  : {acc_non_af:.2f}%")
    print("="*60)
    print(f"🏆 ĐỘ CHÍNH XÁC TỔNG THỂ  : {overall_acc:.2f}%")
    print("="*60 + "\n")

if __name__ == "__main__":
    set_seed(42)
    dataset_path = "/home/linhhima/PPG_ECG/datasets/z_score_norm/total_mimic_af.npz"
    run_pure_p_wave_classification(dataset_path, num_plots_to_show=1, filter_view="non_af")