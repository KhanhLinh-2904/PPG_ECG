import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
from scipy.fftpack import fft, fftfreq
from scipy.stats import kurtosis
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
# 1. TÌM ĐỈNH R (QRS COMPLEX)
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

def detect_ECG_beats(ecg_signal, fs=125):
    r_indices = pan_tompkins_qrs(ecg_signal, fs=fs)
    return r_indices

# =====================================================================
# 2. CHIẾN THUẬT 2: TRÍCH XUẤT ĐẶC TRƯNG HÌNH THÁI VÀ PHỔ NĂNG LƯỢNG
# =====================================================================
def extract_p_wave_features(signal, r_indices, fs=125):
    """
    Trích xuất vùng Sóng P (450ms -> 50ms trước R).
    Tính Variance, Kurtosis, và Phổ năng lượng (1-3Hz vs 3-10Hz).
    """
    search_start = int(0.45 * fs) 
    search_end = int(0.05 * fs)   
    
    # Lọc Bandpass (0.5 - 15 Hz) để giữ lại cả sóng P và sóng f của AFib
    nyq = 0.5 * fs
    b, a = butter(2, [0.5 / nyq, 15.0 / nyq], btype='band')
    clean_signal = filtfilt(b, a, signal)

    features_list = []
    
    for r in r_indices:
        start_idx = r - search_start
        end_idx = r - search_end
        
        if start_idx >= 0 and end_idx < len(signal):
            p_window = clean_signal[start_idx:end_idx]
            p_detrended = p_window - np.mean(p_window)
            
            # --- ĐẶC TRƯNG HÌNH THÁI (MORPHOLOGY) ---
            var_val = np.var(p_detrended)
            kurt_val = kurtosis(p_detrended)
            
            # --- ĐẶC TRƯNG TẦN SỐ (ENERGY SPECTRUM) ---
            # Áp dụng cửa sổ Hamming để tránh rò rỉ phổ
            windowed = p_detrended * np.hamming(len(p_detrended))
            N = len(windowed)
            freqs = fftfreq(N, 1/fs)[:N//2]
            energy = np.abs(fft(windowed))[:N//2] ** 2
            
            # Tính tổng năng lượng trong các dải tần mục tiêu
            power_1_3Hz = np.sum(energy[(freqs >= 1.0) & (freqs <= 3.0)])
            power_3_10Hz = np.sum(energy[(freqs > 3.0) & (freqs <= 10.0)])
            
            # Tỷ lệ năng lượng (Energy Ratio)
            # Tránh chia cho 0
            if power_1_3Hz == 0: power_1_3Hz = 1e-6 
            energy_ratio = power_3_10Hz / power_1_3Hz
            
            features_list.append({
                'waveform': p_detrended,
                'variance': var_val,
                'kurtosis': kurt_val,
                'freqs': freqs,
                'energy': energy,
                'energy_ratio': energy_ratio,
                'start_idx': start_idx,
                'end_idx': end_idx
            })

    return features_list

# =====================================================================
# 3. LUẬT CHẨN ĐOÁN DỰA TRÊN ĐẶC TRƯNG
# =====================================================================
def predict_af_strategy_2(features_list, ratio_threshold=1.5):
    """
    Nếu Tỷ lệ năng lượng (3-10Hz / 1-3Hz) trung bình của các nhịp > threshold -> AFib.
    """
    if len(features_list) == 0:
        return 0, 0.0 # Mặc định Non-AF nếu không có dữ liệu
        
    # Tính trung bình Energy Ratio của tất cả các sóng P trong 10 giây
    avg_ratio = np.mean([f['energy_ratio'] for f in features_list])
    
    # Dự đoán (AF = 1, Non-AF = 0)
    prediction = 1 if avg_ratio > ratio_threshold else 0
    
    return prediction, avg_ratio

# =====================================================================
# 4. HÀM ĐÁNH GIÁ, VẼ ĐỒ THỊ & TỔNG KẾT
# =====================================================================
def evaluate_and_visualize_strategy_2(dataset_path, num_records=300, show_plots=False):
    if not os.path.exists(dataset_path):
        print(f"[!] Không tìm thấy file {dataset_path}.")
        return

    demo_data = np.load(dataset_path, allow_pickle=True)
    demo_ecgs = demo_data["ecgs"]
    demo_labels = demo_data["labels"]
    
    print("\n" + "="*70)
    print(" BẢNG ĐÁNH GIÁ CHIẾN THUẬT 2: PHỔ NĂNG LƯỢNG VÙNG SÓNG P")
    print("="*70)
    print(f"{'Bản ghi':<10} | {'Thực Tế':<10} | {'Dự Đoán':<10} | {'Tỷ lệ Năng lượng':<18} | {'Kết quả'}")
    print("-" * 70)
    
    correct_af, correct_non_af = 0, 0
    total_af, total_non_af = 0, 0

    limit = min(num_records, len(demo_ecgs))
    for i in range(limit):
        sig = demo_ecgs[i]
        true_label = demo_labels[i]
        true_str = "AF" if true_label == 1 else "Non-AF"
        
        r_idx = detect_ECG_beats(sig, fs=125)
        features = extract_p_wave_features(sig, r_idx, fs=125)
        
        # Dự đoán: Mức ngưỡng Ratio = 1.2 (Có thể tinh chỉnh dựa trên dataset của bạn)
        pred_label, avg_ratio = predict_af_strategy_2(features, ratio_threshold=1.2)
        pred_str = "AF" if pred_label == 1 else "Non-AF"
        
        if true_label == 1:
            total_af += 1
            if pred_label == 1: correct_af += 1
        else:
            total_non_af += 1
            if pred_label == 0: correct_non_af += 1
                
        match_marker = "✅ ĐÚNG" if true_label == pred_label else "❌ SAI"
        print(f"Record {i:<3} | {true_str:<10} | {pred_str:<10} | {avg_ratio:<18.3f} | {match_marker}")
        
        # --- VẼ BẢNG ĐIỀU KHIỂN (Nếu bật) ---
        if show_plots and len(features) > 0:
            fig, axs = plt.subplots(3, 1, figsize=(14, 10))
            time_ax = np.arange(len(sig)) / 125
            title_color = "darkred" if pred_label == 1 else "darkgreen"
            
            # ĐỒ THỊ 1: TÍN HIỆU GỐC
            axs[0].plot(time_ax, sig, color='black', linewidth=1)
            axs[0].scatter(r_idx/125, sig[r_idx], color='red', marker='v', zorder=3, label='Đỉnh R')
            for f in features:
                axs[0].axvspan(f['start_idx']/125, f['end_idx']/125, color='gray', alpha=0.2)
            axs[0].set_title(f"Record {i} | Thực tế: [{true_str}] | Máy dự đoán: [{pred_str}]", fontweight='bold', color=title_color)
            axs[0].legend(loc="upper right")
            
            # ĐỒ THỊ 2: OVERLAID P-WAVES (HÌNH THÁI)
            axs[1].set_title(f"Hình thái (Morphology) của {len(features)} vùng trước QRS", fontweight='bold', color='darkblue')
            colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
            for idx, f in enumerate(features):
                t_wave = np.arange(len(f['waveform'])) / 125
                axs[1].plot(t_wave, f['waveform'], color=colors[idx], alpha=0.7)
            axs[1].axhline(0, color='black', linestyle='--', alpha=0.5)
            
            # ĐỒ THỊ 3: PHỔ NĂNG LƯỢNG TRUNG BÌNH
            axs[2].set_title(f"Phổ Năng Lượng Trung Bình | Tỷ lệ (3-10Hz / 1-3Hz) = {avg_ratio:.3f}", fontweight='bold', color='darkmagenta')
            avg_energy = np.mean([f['energy'] for f in features], axis=0)
            freqs = features[0]['freqs']
            
            axs[2].plot(freqs, avg_energy, color='purple', linewidth=2)
            axs[2].axvspan(1.0, 3.0, color='green', alpha=0.15, label='Sóng P (1-3 Hz)')
            axs[2].axvspan(3.0, 10.0, color='red', alpha=0.15, label='F-waves Rung Nhĩ (3-10 Hz)')
            axs[2].set_xlim(0, 15)
            axs[2].legend(loc="upper right")
            
            plt.tight_layout()
            plt.show(block=True)

    # =====================================================================
    # IN BẢNG TỔNG KẾT
    # =====================================================================
    print("\n" + "="*60)
    print(" 📊 TỔNG KẾT KẾT QUẢ - CHIẾN THUẬT 2")
    print("="*60)
    acc_af = (correct_af / total_af * 100) if total_af > 0 else 0
    acc_non_af = (correct_non_af / total_non_af * 100) if total_non_af > 0 else 0
    total_records = total_af + total_non_af
    overall_acc = ((correct_af + correct_non_af) / total_records * 100) if total_records > 0 else 0

    print(f"🔸 Tổng số bản ghi đã test : {total_records} records")
    print("-" * 60)
    print(f"🔴 RUNG NHĨ (AF):")
    print(f"   - Số ca thực tế         : {total_af} ca")
    print(f"   - Máy dự đoán ĐÚNG      : {correct_af} ca")
    print(f"   => Độ chính xác nhóm AF : {acc_af:.2f}%")
    print("-" * 60)
    print(f"🟢 BÌNH THƯỜNG (Non-AF):")
    print(f"   - Số ca thực tế         : {total_non_af} ca")
    print(f"   - Máy dự đoán ĐÚNG      : {correct_non_af} ca")
    print(f"   => Độ chính xác Non-AF  : {acc_non_af:.2f}%")
    print("="*60)
    print(f"🏆 ĐỘ CHÍNH XÁC TỔNG THỂ  : {overall_acc:.2f}%")
    print("="*60 + "\n")

if __name__ == "__main__":
    set_seed(42)
    dataset_path = "/home/linhhima/PPG_ECG/datasets/z_score_norm/MIT_BIH_test_segments.npz"
    
    # Bật show_plots=True để xem Bảng điều khiển phân tích Tần số cực kỳ chi tiết
    evaluate_and_visualize_strategy_2(dataset_path, num_records=20000, show_plots=False)