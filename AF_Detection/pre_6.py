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
# 2. PHÂN TÍCH VÙNG SÓNG P & LƯU TỌA ĐỘ ĐỈNH P
# =====================================================================
def analyze_p_wave_for_af(signal, fs=125):
    r_indices = pan_tompkins_qrs(signal, fs)
    
    if len(r_indices) < 2:
        return 0.0, 0.0, [], []  # Thêm trả về list rỗng cho r_indices và p_peaks
        
    search_start = int(0.45 * fs) 
    search_end = int(0.05 * fs)   
    
    nyq = 0.5 * fs
    b, a = butter(2, [0.5 / nyq, 15.0 / nyq], btype='band')
    clean_signal = filtfilt(b, a, signal)

    num_p_found = 0
    energy_ratios = []
    p_peaks_indices = [] # Mảng lưu tọa độ đỉnh P
    
    for r in r_indices:
        start_idx = r - search_start
        end_idx = r - search_end
        
        if start_idx >= 0 and end_idx < len(signal):
            p_window = clean_signal[start_idx:end_idx]
            p_detrended = p_window - np.mean(p_window)
            
            # --- KIỂM TRA SỰ HIỆN DIỆN CỦA SÓNG P ---
            peaks, _ = find_peaks(p_detrended, prominence=0.5)
            if len(peaks) > 0:
                num_p_found += 1
                
                # CẬP NHẬT: Lấy đỉnh cao nhất trong cửa sổ làm P-peak và lưu lại tọa độ
                peak_idx_local = peaks[np.argmax(p_detrended[peaks])]
                abs_p_peak = start_idx + peak_idx_local
                p_peaks_indices.append(abs_p_peak)
                
            # --- TÍNH PHỔ NĂNG LƯỢNG ---
            windowed = p_detrended * np.hamming(len(p_detrended))

            N = len(windowed)
            freqs = fftfreq(N, 1/fs)[:N//2]
            energy = np.abs(fft(windowed))[:N//2] ** 2
            
            power_1_3Hz = np.sum(energy[(freqs >= 1.0) & (freqs < 4.0)])
            power_3_10Hz = np.sum(energy[(freqs >= 4.0) & (freqs <= 100.0)])
            
            if power_1_3Hz == 0: power_1_3Hz = 1e-6 
            energy_ratios.append(power_3_10Hz / power_1_3Hz)

    p_ratio = num_p_found / len(r_indices)
    avg_energy_ratio = np.mean(energy_ratios) if len(energy_ratios) > 0 else 0
    
    return p_ratio, avg_energy_ratio, p_peaks_indices, r_indices

# =====================================================================
# 3. CHẨN ĐOÁN & KIỂM THỬ
# =====================================================================
def run_pure_p_wave_classification(dataset_path, num_plots_to_show=5):
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
    ENERGY_THRESH = 2.0

    print("\n[*] Đang tiến hành phân tích Sóng P...")
    
    plots_shown = 0 # Biến đếm số lượng đồ thị đã vẽ

    for i in range(len(ecgs)):
        sig = ecgs[i]
        true_lbl = labels[i]
        # if true_lbl == 1:
        # Nhận thêm p_peaks_indices và r_indices từ hàm
        p_ratio, energy_ratio, p_peaks_idx, r_idx = analyze_p_wave_for_af(sig, fs=125)
        
        # LUẬT CHẨN ĐOÁN
        if p_ratio < P_RATIO_THRESH or energy_ratio > ENERGY_THRESH:
            pred_lbl = 1 # AF
        else:
            pred_lbl = 0 # Non-AF
            
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
        if plots_shown < num_plots_to_show and len(r_idx) > 0:
            fig, ax = plt.subplots(1, 1, figsize=(12, 4))
            time_ax = np.arange(len(sig)) / 125
            
            # Cài đặt màu tiêu đề: Đúng (Xanh), Sai (Đỏ)
            title_color = 'darkgreen' if true_lbl == pred_lbl else 'darkred'
            true_str = "AF" if true_lbl == 1 else "Non-AF"
            pred_str = "AF" if pred_lbl == 1 else "Non-AF"
            
            # Vẽ tín hiệu gốc
            ax.plot(time_ax, sig, color='black', linewidth=1, alpha=0.8, label='ECG Signal')
            
            # Vẽ đỉnh R (Màu đỏ)
            ax.scatter(r_idx/125, sig[r_idx], color='red', marker='v', s=60, zorder=3, label='Đỉnh R')
            
            # Vẽ đỉnh P (Màu xanh lá) - Cần ép kiểu sang numpy array trước khi chia cho 125
            if len(p_peaks_idx) > 0:
                p_peaks_arr = np.array(p_peaks_idx)
                ax.scatter(p_peaks_arr/125, sig[p_peaks_arr], color='limegreen', marker='o', s=50, zorder=4, label='Đỉnh P')

            # Highlight không gian tìm kiếm (tùy chọn)
            for r in r_idx:
                s_time = max(0, (r - int(0.45 * 125)) / 125)
                e_time = max(0, (r - int(0.05 * 125)) / 125)
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
    dataset_path = "/home/linhhima/PPG_ECG/AF_Detection/total_mimic_af_recon.npz"
    # Bạn có thể đổi num_plots_to_show thành 0 nếu chỉ muốn xem bảng tổng kết cho nhanh
    run_pure_p_wave_classification(dataset_path, num_plots_to_show=1)