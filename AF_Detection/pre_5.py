import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
from scipy.fftpack import fft, fftfreq
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

# =====================================================================
# 1. TIỀN XỬ LÝ & TÌM ĐỈNH R (PAN-TOMPKINS)
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
# 2. TRÍCH XUẤT ĐẶC TRƯNG TỪ VÙNG SÓNG P (P-WAVE WINDOW)
# =====================================================================
def extract_p_wave_features(signal, r_indices, fs=125):
    """
    Trích xuất vùng không gian trước QRS và tính toán các đặc trưng sống còn.
    """
    search_start = int(0.45 * fs) 
    search_end = int(0.05 * fs)   
    
    # Lọc Bandpass (0.5 - 15 Hz) để giữ lại P-wave và f-wave
    nyq = 0.5 * fs
    b, a = butter(2, [0.5 / nyq, 15.0 / nyq], btype='band')
    clean_signal = filtfilt(b, a, signal)

    num_p_found = 0
    energy_ratios = []
    valid_windows = []
    
    for r in r_indices:
        start_idx = r - search_start
        end_idx = r - search_end
        
        if start_idx >= 0 and end_idx < len(signal):
            p_window = clean_signal[start_idx:end_idx]
            p_detrended = p_window - np.mean(p_window)
            valid_windows.append((start_idx, end_idx))
            
            # 1. TÌM SỰ TỒN TẠI CỦA SÓNG P (PEAK DETECTION)
            peaks, _ = find_peaks(p_detrended, prominence=0.005)
            if len(peaks) > 0:
                num_p_found += 1
                
            # 2. PHÂN TÍCH NĂNG LƯỢNG TẦN SỐ (FFT)
            windowed = p_detrended * np.hamming(len(p_detrended))
            N = len(windowed)
            freqs = fftfreq(N, 1/fs)[:N//2]
            energy = np.abs(fft(windowed))[:N//2] ** 2
            
            power_1_3Hz = np.sum(energy[(freqs >= 1.0) & (freqs <= 3.0)])
            power_3_10Hz = np.sum(energy[(freqs > 3.0) & (freqs <= 10.0)])
            
            if power_1_3Hz == 0: power_1_3Hz = 1e-6 
            energy_ratios.append(power_3_10Hz / power_1_3Hz)

    # Tính toán đặc trưng tổng thể cho cả đoạn record
    num_r = len(r_indices)
    p_ratio = (num_p_found / num_r) if num_r > 0 else 0
    avg_energy_ratio = np.mean(energy_ratios) if len(energy_ratios) > 0 else 0
    
    return {
        'p_ratio': p_ratio,
        'avg_energy_ratio': avg_energy_ratio,
        'num_p': num_p_found,
        'num_r': num_r,
        'windows': valid_windows
    }

# =====================================================================
# 3. THUẬT TOÁN PHÂN LOẠI LAI (HYBRID CLASSIFIER)
# =====================================================================
def classify_afib(features, p_ratio_thresh=0.5, energy_ratio_thresh=1.2):
    """
    Chẩn đoán AFib dựa trên kết hợp 2 quy tắc: Mất sóng P và Năng lượng rung giật.
    """
    p_ratio = features['p_ratio']
    energy_ratio = features['avg_energy_ratio']
    
    # LUẬT KẾT HỢP (HYBRID RULE)
    # Nếu mất quá nửa số sóng P HOẶC năng lượng tần số rung nhĩ quá cao
    if p_ratio < p_ratio_thresh or energy_ratio > energy_ratio_thresh:
        return 1 # AFib
    else:
        return 0 # Non-AFib

# =====================================================================
# 4. HÀM CHẠY KIỂM CHỨNG & VẼ BIỂU ĐỒ (DASHBOARD)
# =====================================================================
def run_af_classification_pipeline(dataset_path, num_records=300, show_plots=False):
    if not os.path.exists(dataset_path):
        print(f"[!] Lỗi: Không tìm thấy file {dataset_path}")
        return

    data = np.load(dataset_path, allow_pickle=True)
    ecgs = data["ecgs"]
    labels = data["labels"]
    
    print("\n" + "="*80)
    print(" 🚀 HỆ THỐNG CHẨN ĐOÁN RUNG NHĨ DỰA TRÊN P-WAVE (HYBRID MODEL)")
    print("="*80)
    print(f"{'Bản ghi':<8} | {'Thực Tế':<8} | {'Dự Đoán':<8} | {'P-Ratio':<10} | {'Energy-Ratio':<12} | {'Kết Quả'}")
    print("-" * 80)
    
    correct_af, correct_non_af, total_af, total_non_af = 0, 0, 0, 0
    limit = min(num_records, len(ecgs))
    
    for i in range(limit):
        sig = ecgs[i]
        true_lbl = labels[i]
        true_str = "AF" if true_lbl == 1 else "Non-AF"
        
        # 1. Tìm đỉnh R
        r_idx = pan_tompkins_qrs(sig, fs=125)
        
        if len(r_idx) >= 2:
            # 2. Trích xuất đặc trưng vùng P
            feats = extract_p_wave_features(sig, r_idx, fs=125)
            
            # 3. Phân loại AFib
            pred_lbl = classify_afib(feats)
            pred_str = "AF" if pred_lbl == 1 else "Non-AF"
            
            # 4. Thống kê
            if true_lbl == 1:
                total_af += 1
                if pred_lbl == 1: correct_af += 1
            else:
                total_non_af += 1
                if pred_lbl == 0: correct_non_af += 1
                    
            match = "✅ ĐÚNG" if true_lbl == pred_lbl else "❌ SAI"
            
            print(f"Rec {i:<5} | {true_str:<8} | {pred_str:<8} | {feats['p_ratio']:<10.2f} | {feats['avg_energy_ratio']:<12.3f} | {match}")
            
            # 5. Vẽ đồ thị nếu được yêu cầu
            if show_plots and true_lbl != pred_lbl: # Chỉ vẽ những ca dự đoán sai để phân tích
                fig, ax = plt.subplots(1, 1, figsize=(12, 4))
                t_ax = np.arange(len(sig)) / 125
                ax.plot(t_ax, sig, color='black', linewidth=1)
                ax.scatter(r_idx/125, sig[r_idx], color='red', marker='v', zorder=3)
                
                for (s, e) in feats['windows']:
                    ax.axvspan(s/125, e/125, color='gold', alpha=0.3)
                    
                ax.set_title(f"Record {i} | Thực tế: {true_str} | Dự đoán: {pred_str}\nP-Ratio: {feats['p_ratio']:.2f} | Energy Ratio: {feats['avg_energy_ratio']:.2f}", color='darkred')
                plt.tight_layout()
                plt.show(block=True)
        else:
            print(f"Rec {i:<5} | Tín hiệu lỗi / Quá ngắn")

    # =====================================================================
    # BẢNG TỔNG KẾT
    # =====================================================================
    print("\n" + "="*80)
    print(" 📊 TỔNG KẾT HIỆU SUẤT (PERFORMANCE SUMMARY)")
    print("="*80)
    acc_af = (correct_af / total_af * 100) if total_af > 0 else 0
    acc_non_af = (correct_non_af / total_non_af * 100) if total_non_af > 0 else 0
    total_recs = total_af + total_non_af
    overall_acc = ((correct_af + correct_non_af) / total_recs * 100) if total_recs > 0 else 0

    print(f"Tổng số bản ghi test : {total_recs}")
    print(f"Độ chính xác AF (Sensitivity) : {acc_af:.2f}% ({correct_af}/{total_af})")
    print(f"Độ chính xác Non-AF (Specificity): {acc_non_af:.2f}% ({correct_non_af}/{total_non_af})")
    print(f"🏆 ĐỘ CHÍNH XÁC TỔNG THỂ (ACC) : {overall_acc:.2f}%")
    print("="*80 + "\n")

if __name__ == "__main__":
    set_seed(42)
    dataset_path = "/home/linhhima/PPG_ECG/AF_Detection/total_mimic_af_recon.npz"
    
    # Mẹo: Để `show_plots=True` nếu bạn muốn xem lại biểu đồ của những ca máy tính dự đoán SAI (False Positives / False Negatives).
    run_af_classification_pipeline(dataset_path, num_records=20000, show_plots=False)