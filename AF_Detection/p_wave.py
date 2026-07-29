import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
from scipy.fftpack import fft, fftfreq
import random
import torch
from tqdm import tqdm
import tkinter as tk

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Hide the main Tkinter window
root = tk.Tk()
root.withdraw()

# =====================================================================
# 1. PREPROCESSING & PEAK DETECTION
# =====================================================================
def pan_tompkins_qrs(ecg_signal: np.ndarray, fs: int = 125, external_threshold: float = None, return_threshold: bool = False):
    ecg_signal = np.array(ecg_signal).flatten()
    nyq = 0.5 * fs
    b, a = butter(1, [5.0 / nyq, 15.0 / nyq], btype='band')
    filtered_ecg = filtfilt(b, a, ecg_signal)
    
    diff_ecg = np.diff(filtered_ecg)
    diff_ecg = np.insert(diff_ecg, 0, diff_ecg[0])
    squared_ecg = diff_ecg ** 2

    window_width = int(0.15 * fs)
    integrated_ecg = np.convolve(squared_ecg, np.ones(window_width) / window_width, mode='same')
    
    if external_threshold is not None:
        threshold = external_threshold
    else:
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
            
    if return_threshold:
        return np.array(r_peaks), threshold
    return np.array(r_peaks)

def analyze_p_wave_for_af(signal, fs=125, gt_threshold=None):
    r_indices = pan_tompkins_qrs(signal, fs, external_threshold=gt_threshold)
    
    if len(r_indices) < 2:
        return 0.0, 0.0, [], []  
        
    nyq = 0.5 * fs
    b, a = butter(2, [0.5 / nyq, 15.0 / nyq], btype='band')
    clean_signal = filtfilt(b, a, signal)

    num_p_found = 0
    p_peaks_indices = []
    
    for i in range(1, len(r_indices)):
        r = r_indices[i]
        r_prev = r_indices[i-1] 
        rr_distance = r - r_prev
        
        search_start = int(rr_distance * 0.5)
        search_end = int(0.05 * rr_distance)
        
        start_idx = r - search_start
        end_idx = r - search_end
        
        if start_idx >= r_prev and end_idx < len(signal):
            p_window = clean_signal[start_idx:end_idx]
            p_detrended = p_window - np.mean(p_window)
            
            peaks, _ = find_peaks(p_detrended, prominence=0.5)
            if len(peaks) > 0:
                num_p_found += 1
                peak_idx_local = peaks[np.argmax(p_detrended[peaks])]
                p_peaks_indices.append(start_idx + peak_idx_local)

    valid_beats = len(r_indices) - 1
    p_ratio = num_p_found / valid_beats if valid_beats > 0 else 0
    
    return p_ratio, 0.0, p_peaks_indices, r_indices

# =====================================================================
# 2. FEATURE EXTRACTION PIPELINE 
# =====================================================================
def load_and_extract_p_ratios(dataset_path, dataset_origin_path, dataset_name, sample_ratio=0.5, fs=125):
    print(f"\n[*] Loading {dataset_name} from {dataset_path}...")
    data = np.load(dataset_path, allow_pickle=True)
    ecgs = data["ecgs"]
    labels = data["labels"]
    
    data_orig = np.load(dataset_origin_path, allow_pickle=True)
    ecgs_orig = data_orig["ecgs"]

    num_samples = len(ecgs)
    indices = np.random.permutation(num_samples)
    target_count = int(num_samples * sample_ratio)
    selected_indices = indices[:target_count]
    
    print(f"[*] Selected {target_count} / {num_samples} records ({sample_ratio*100}% random split).")

    p_ratios = []
    true_labels = []

    for i in tqdm(selected_indices, desc=f"Extracting P-Ratios ({dataset_name})"):
        sig = ecgs[i]
        true_sig = ecgs_orig[i]
        true_lbl = labels[i]

        _, gt_threshold = pan_tompkins_qrs(true_sig, fs=fs, return_threshold=True)
        p_ratio, _, _, _ = analyze_p_wave_for_af(sig, fs=fs, gt_threshold=gt_threshold)

        p_ratios.append(p_ratio)
        true_labels.append(true_lbl)

    return np.array(p_ratios), np.array(true_labels)

# =====================================================================
# 3. GRID SEARCH: LEARN THRESHOLD 
# =====================================================================
def learn_best_threshold(p_ratios, true_labels):
    # Sử dụng np.arange để tạo đúng 50 mức Threshold (từ 0.02 đến 1.0 với bước nhảy 0.02)
    thresholds = np.arange(0.02, 1.01, 0.02)
    # Làm tròn để tránh sai số thập phân của máy tính
    thresholds = np.round(thresholds, 2)
    
    best_thresh = 0.02 # Khởi tạo bằng giá trị đầu tiên của mảng
    best_acc = 0.0
    all_accuracies = [] # Lưu lại độ chính xác để vẽ biểu đồ

    print("\n" + "="*60)
    print(" 🔍 GRID SEARCH: LEARNING BEST THRESHOLD (MIT-BIH 50%)")
    print("="*60)
    
    for thresh in thresholds:
        # Nếu tỷ lệ P/R < threshold, dự đoán là AF (1), ngược lại là Non-AF (0)
        preds = (p_ratios < thresh).astype(int)
        acc = np.mean(preds == true_labels)
        all_accuracies.append(acc)
        
        # Cập nhật threshold tốt nhất
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
            
        # (Tùy chọn) In quá trình dò tìm để dễ theo dõi
        if int(thresh * 100) % 10 == 0: 
            print(f"Thử nghiệm Threshold = {thresh:.2f} | Accuracy = {acc*100:.2f}%")

    print("="*60)
    print(f"🌟 KẾT QUẢ TỐT NHẤT: BEST THRESHOLD = {best_thresh:.2f} (Accuracy: {best_acc*100:.2f}%)")
    print("="*60)
    
    return best_thresh, thresholds, np.array(all_accuracies)

# =====================================================================
# KHỐI LỆNH MỚI: VẼ BIỂU ĐỒ ĐÁNH GIÁ (VISUALIZATION)
# =====================================================================
def plot_threshold_analysis(p_ratios, true_labels, thresholds, accuracies, best_thresh):
    print("[*] Generating Threshold Analysis Plots...")
    plt.figure(figsize=(15, 6))

    # --- Subplot 1: Đường cong Grid Search ---
    plt.subplot(1, 2, 1)
    plt.plot(thresholds, accuracies * 100, marker='o', linestyle='-', color='dodgerblue', markersize=5)
    plt.axvline(x=best_thresh, color='red', linestyle='--', linewidth=2, label=f'Best Threshold: {best_thresh:.2f}')
    
    plt.title("Grid Search: Accuracy vs. Threshold", fontsize=14, fontweight='bold')
    plt.xlabel("P-over-R Ratio Threshold", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)

    # --- Subplot 2: Phân phối của tập dữ liệu (Histogram) ---
    plt.subplot(1, 2, 2)
    af_ratios = p_ratios[true_labels == 1]
    non_af_ratios = p_ratios[true_labels == 0]

    # Vẽ histogram cho AF và Non-AF
    plt.hist(non_af_ratios, bins=30, alpha=0.6, color='mediumseagreen', edgecolor='black', label='Non-AF (Normal)')
    plt.hist(af_ratios, bins=30, alpha=0.6, color='crimson', edgecolor='black', label='AF (Atrial Fibrillation)')
    
    # Vẽ vạch ranh giới quyết định
    plt.axvline(x=best_thresh, color='black', linestyle='-.', linewidth=2.5, label=f'Decision Boundary ({best_thresh:.2f})')

    plt.title("P/R Ratio Distribution (Train Set)", fontsize=14, fontweight='bold')
    plt.xlabel("P-over-R Ratio Value", fontsize=12)
    plt.ylabel("Number of Segments", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)

    plt.tight_layout()
    plt.show(block=True)

# =====================================================================
# 4. EVALUATION: TEST THRESHOLD (Trên MIMIC AF)
# =====================================================================
def test_learned_threshold(learned_thresh, p_ratios, true_labels):
    print("\n" + "="*60)
    print(" 📊 THỬ NGHIỆM ĐÁNH GIÁ CHÉO (TEST ON 50% MIMIC AF)")
    print("="*60)
    print(f"[*] Đang áp dụng Learned Threshold = {learned_thresh:.4f}...")

    preds = (p_ratios < learned_thresh).astype(int)

    total_af = np.sum(true_labels == 1)
    total_non_af = np.sum(true_labels == 0)

    correct_af = np.sum((preds == 1) & (true_labels == 1))
    correct_non_af = np.sum((preds == 0) & (true_labels == 0))

    acc_af = (correct_af / total_af * 100) if total_af > 0 else 0
    acc_non_af = (correct_non_af / total_non_af * 100) if total_non_af > 0 else 0
    overall_acc = np.mean(preds == true_labels) * 100

    print(f"🔹 Tổng số bản ghi Test       : {len(true_labels)} records")
    print("-" * 60)
    print(f"🔴 RUNG NHĨ (AF):")
    print(f"   - Số ca thực tế            : {total_af}")
    print(f"   - Dự đoán ĐÚNG             : {correct_af}")
    print(f"   => Độ nhạy (Sensitivity)   : {acc_af:.2f}%")
    print("-" * 60)
    print(f"🟢 BÌNH THƯỜNG (Non-AF):")
    print(f"   - Số ca thực tế            : {total_non_af}")
    print(f"   - Dự đoán ĐÚNG             : {correct_non_af}")
    print(f"   => Đặc hiệu (Specificity)  : {acc_non_af:.2f}%")
    print("="*60)
    print(f"🏆 ĐỘ CHÍNH XÁC CHUNG (ACC)   : {overall_acc:.2f}%")
    print("="*60 + "\n")


# =====================================================================
# MAIN PIPELINE
# =====================================================================
if __name__ == "__main__":
    set_seed(42)
    
    MIT_BIH_PATH = "/home/linhhima/PPG_ECG/datasets/z_score_norm/MIT_BIH_train_segments.npz"
    MIMIC_AF_PATH = "/home/linhhima/PPG_ECG/datasets/z_score_norm/total_mimic_af.npz"
    
    # BƯỚC 1: Load 50% MIT-BIH để HỌC
    train_p_ratios, train_labels = load_and_extract_p_ratios(
        dataset_path=MIT_BIH_PATH, 
        dataset_origin_path=MIT_BIH_PATH, 
        dataset_name="MIT-BIH Train",
        sample_ratio=0.50,
        fs=250
    )
    
    # Tìm Threshold tốt nhất và lấy thêm dữ liệu để vẽ
    best_threshold, threshold_list, accuracy_list = learn_best_threshold(train_p_ratios, train_labels)
    
    # Vẽ đồ thị hiển thị kết quả Grid Search
    plot_threshold_analysis(train_p_ratios, train_labels, threshold_list, accuracy_list, best_threshold)
    
    # BƯỚC 2: Load 50% MIMIC AF để KIỂM THỬ
    test_p_ratios, test_labels = load_and_extract_p_ratios(
        dataset_path=MIMIC_AF_PATH, 
        dataset_origin_path=MIMIC_AF_PATH, 
        dataset_name="MIMIC AF Test",
        sample_ratio=0.50,
        fs=125
    )
    
    # Áp dụng Threshold vừa học được vào tập MIMIC AF
    test_learned_threshold(best_threshold, test_p_ratios, test_labels)