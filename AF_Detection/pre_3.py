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

def detect_ECG_beats_with_indices(ecg_signal):
    fs = 125
    r_indices = pan_tompkins_qrs(ecg_signal, fs=fs)
    if len(r_indices) < 2:
        return np.array([]), np.array([])
    rr_intervals = np.diff(r_indices) / fs
    return rr_intervals, r_indices

# =====================================================================
# 2. TÌM SÓNG P (P-WAVE DETECTION)
# =====================================================================
def detect_p_wave_boundaries(signal, r_indices, fs=125):
    p_data = []
    search_start = int(0.45 * fs) 
    search_end = int(0.05 * fs)   
    
    nyq = 0.5 * fs
    b, a = butter(2, [0.5 / nyq, 10.0 / nyq], btype='band')
    clean_signal = filtfilt(b, a, signal)

    for r in r_indices:
        start_idx = r - search_start
        end_idx = r - search_end
        
        if start_idx >= 0 and end_idx < len(signal):
            search_window = clean_signal[start_idx:end_idx]
            detrended_window = search_window - np.mean(search_window)
            
            # Chỉ coi là sóng P nếu có đỉnh lồi lên rõ ràng
            peaks, _ = find_peaks(detrended_window, prominence=0.005)
            
            if len(peaks) > 0:
                peak_idx_local = peaks[np.argmax(detrended_window[peaks])]
                peak_val = detrended_window[peak_idx_local]
                
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
                
                p_data.append({
                    'onset': abs_onset,
                    'peak': abs_peak,
                    'endset': abs_endset
                })

    return p_data

# =====================================================================
# 3. CHIẾN THUẬT 1: CHẨN ĐOÁN DỰA VÀO TỶ LỆ P-WAVE
# =====================================================================
def predict_af_strategy_1(signal, r_indices, fs=125, af_threshold=0.9):
    """
    Tính toán tỷ lệ Sóng P tìm được / Số đỉnh R.
    Nếu tỷ lệ < af_threshold (Mặc định 0.5), hệ thống dự đoán là Rung Nhĩ (1).
    """
    if len(r_indices) == 0:
        return 0, 0.0, 0, 0  # Trả về mặc định nếu tín hiệu lỗi
    
    p_data = detect_p_wave_boundaries(signal, r_indices, fs)
    
    num_r = len(r_indices)
    num_p = len(p_data)
    
    # Tính tỷ lệ
    p_ratio = num_p / num_r
    
    # Luật chẩn đoán
    prediction = 1 if p_ratio < af_threshold else 0
    
    return prediction, p_ratio, num_p, num_r, p_data

# =====================================================================
# 4. HÀM ĐÁNH GIÁ, VẼ ĐỒ THỊ VÀ TỔNG KẾT
# =====================================================================
def evaluate_and_visualize_strategy_1(dataset_path, num_records=300, show_plots=False):
    if not os.path.exists(dataset_path):
        print(f"[!] Không tìm thấy file {dataset_path}.")
        return

    demo_data = np.load(dataset_path, allow_pickle=True)
    demo_ecgs = demo_data["ecgs"]
    demo_labels = demo_data["labels"]
    
    print("\n" + "="*60)
    print(" BẢNG ĐÁNH GIÁ CHIẾN THUẬT 1: SỰ VẮNG MẶT CỦA SÓNG P")
    print("="*60)
    print(f"{'Bản ghi':<10} | {'Thực Tế':<10} | {'Dự Đoán':<10} | {'Số P / R':<12} | {'Tỷ Lệ':<8} | {'Kết quả'}")
    print("-" * 60)
    
    # --- KHỞI TẠO CÁC BIẾN TỔNG KẾT ---
    correct_af = 0
    correct_non_af = 0
    total_af = 0
    total_non_af = 0

    # Chạy vòng lặp qua số lượng bản ghi được chỉ định
    limit = min(num_records, len(demo_ecgs))
    for i in range(limit):
        sig = demo_ecgs[i]
        true_label = demo_labels[i]
        true_str = "AF" if true_label == 1 else "Non-AF"
        
        _, r_idx = detect_ECG_beats_with_indices(sig)
        
        # Gọi thuật toán chẩn đoán
        pred_label, p_ratio, num_p, num_r, p_data = predict_af_strategy_1(sig, r_idx, fs=125, af_threshold=0.92)
        pred_str = "AF" if pred_label == 1 else "Non-AF"
        
        # --- CẬP NHẬT BIẾN ĐẾM TỔNG KẾT ---
        if true_label == 1:
            total_af += 1
            if pred_label == 1:
                correct_af += 1
        else:
            total_non_af += 1
            if pred_label == 0:
                correct_non_af += 1
                
        # Đánh dấu đúng/sai cho log
        match_marker = "✅ ĐÚNG" if true_label == pred_label else "❌ SAI"
        
        print(f"Record {i:<3} | {true_str:<10} | {pred_str:<10} | {num_p:>3} / {num_r:<6} | {p_ratio:.2f}     | {match_marker}")
        
        # HIỂN THỊ ĐỒ THỊ (NẾU ĐƯỢC BẬT)
        if show_plots and len(r_idx) > 0:
            fig, ax = plt.subplots(1, 1, figsize=(12, 5))
            time_ax = np.arange(len(sig)) / 125
            
            title_color = "darkred" if pred_label == 1 else "darkgreen"
            bg_color = "#ffe6e6" if pred_label == 1 else "#e6ffe6"
            ax.set_facecolor(bg_color)
            
            ax.plot(time_ax, sig, color='black', linewidth=1)
            ax.scatter(r_idx/125, sig[r_idx], color='red', marker='v', s=60, zorder=3, label='Đỉnh R')
            
            for p in p_data:
                ax.axvspan(p['onset']/125, p['endset']/125, color='gold', alpha=0.5)
                ax.scatter(p['peak']/125, sig[p['peak']], color='blue', marker='o', s=30, zorder=4)
            
            if len(p_data) > 0:
                ax.scatter([], [], color='gold', alpha=0.5, label='Sóng P (P-wave)')
                ax.scatter([], [], color='blue', marker='o', label='Đỉnh P (P-peak)')

            title_str = (f"Record {i} | Thực tế: [{true_str}] | Máy dự đoán: [{pred_str}] {match_marker}\n"
                         f"Tìm thấy {num_p} Sóng P / {num_r} Nhịp tim (Tỷ lệ: {p_ratio:.2f})")
            ax.set_title(title_str, fontweight='bold', color=title_color)
            ax.set_xlabel("Thời gian (giây)")
            ax.set_ylabel("Amplitude")
            ax.legend(loc="upper right")
            
            plt.tight_layout()
            plt.show(block=True)

    # =====================================================================
    # IN BẢNG TỔNG KẾT (SUMMARY REPORT) CỦA TOÀN BỘ QUÁ TRÌNH RUN
    # =====================================================================
    print("\n" + "="*60)
    print(" 📊 TỔNG KẾT KẾT QUẢ DỰ ĐOÁN (SUMMARY REPORT)")
    print("="*60)
    
    # Xử lý tránh chia cho 0 nếu tập dữ liệu bị lệch (không có nhãn nào đó)
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
    print(f"🏆 ĐỘ CHÍNH XÁC TỔNG THỂ (OVERALL ACCURACY): {overall_acc:.2f}%")
    print("="*60 + "\n")

if __name__ == "__main__":
    set_seed(42)
    dataset_path = "/home/linhhima/PPG_ECG/datasets/z_score_norm/MIT_BIH_test_segments.npz"
    
    # Chạy 300 bản ghi. 
    # Bật show_plots=False để chạy lướt nhanh và xem bảng tổng kết ở cuối cùng.
    # Bật show_plots=True nếu bạn muốn soi lại từng biểu đồ.
    evaluate_and_visualize_strategy_1(dataset_path, num_records=20000, show_plots=False)