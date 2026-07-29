import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
import random
from tqdm import tqdm

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

# =====================================================================
# 1. PREPROCESSING & PEAK DETECTION [cite: 458, 1071]
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
# 2. FEATURE EXTRACTION (SINGLE DATASET) [cite: 1168, 1170]
# =====================================================================
def load_and_extract_metrics(dataset_path, dataset_name, sample_ratio=1.0, fs=125):
    print(f"\n[*] Loading dataset: {dataset_name} từ {dataset_path}...")
    data = np.load(dataset_path, allow_pickle=True)
    ecgs = data["ecgs"]
    labels = data["labels"]

    num_samples = len(ecgs)
    target_count = int(num_samples * sample_ratio)
    
    # Lấy mẫu tuần tự hoặc ngẫu nhiên theo tỷ lệ truyền vào
    indices = np.arange(num_samples)
    if sample_ratio < 1.0:
        indices = np.random.permutation(num_samples)[:target_count]
    else:
        print(f"[*] Sử dụng toàn bộ tập dữ liệu test chứa {num_samples} phân đoạn.")

    p_ratios = []
    true_labels = []

    for i in tqdm(indices, desc=f"Đang trích xuất tỷ lệ P/R ({dataset_name})"):
        sig = ecgs[i]
        true_lbl = labels[i]

        # Trích xuất tỷ lệ P/R trực tiếp trên dữ liệu [cite: 1056, 1087]
        _, gt_threshold = pan_tompkins_qrs(sig, fs=fs, return_threshold=True)
        p_ratio, _, _, _ = analyze_p_wave_for_af(sig, fs=fs, gt_threshold=gt_threshold)

        p_ratios.append(p_ratio)
        true_labels.append(true_lbl)

    return np.array(p_ratios), np.array(true_labels)

# =====================================================================
# 3. ROC CURVE & YOUDEN'S INDEX OPTIMIZATION [cite: 1364, 1367]
# =====================================================================
def optimize_via_youden_index(p_ratios, true_labels):
    # Khởi tạo mảng Threshold quét mịn từ 0.0 đến 1.0 với bước nhảy 0.01
    thresholds = np.linspace(0.0, 1.0, 101)
    
    sensitivities = []
    specificities = []
    youdens_indices = []
    accuracies = []

    print("\n" + "="*70)
    print(" 📊 ROC ANALYSIS & YOUDEN'S INDEX OPTIMIZATION (QUÉT THRESHOLD 0 -> 1)")
    print("="*70)

    for thresh in thresholds:
        # Nếu P/R < threshold, dự đoán là AF (1), ngược lại là Non-AF (0) 
        preds = (p_ratios < thresh).astype(int)
        
        # Tính toán các thành phần ma trận nhầm lẫn [cite: 1346]
        tp = np.sum((preds == 1) & (true_labels == 1))
        tn = np.sum((preds == 0) & (true_labels == 0))
        fp = np.sum((preds == 1) & (true_labels == 0))
        fn = np.sum((preds == 0) & (true_labels == 1))

        # Tính toán các chỉ số thống kê y sinh [cite: 1348, 1357, 1361]
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        # Tính toán chỉ số Youden Index: J = Sensitivity + Specificity - 1 
        j_index = sensitivity + specificity - 1.0

        sensitivities.append(sensitivity)
        specificities.append(specificity)
        youdens_indices.append(j_index)
        accuracies.append(accuracy)

    sensitivities = np.array(sensitivities)
    specificities = np.array(specificities)
    youdens_indices = np.array(youdens_indices)
    accuracies = np.array(accuracies)

    # Tìm vị trí đạt chỉ số Youden Index lớn nhất [cite: 1364, 1367]
    best_idx = np.argmax(youdens_indices)
    best_threshold = thresholds[best_idx]
    best_j = youdens_indices[best_idx]
    
    # Khôi phục các chỉ số tại ngưỡng tối ưu nhất [cite: 1791]
    opt_preds = (p_ratios < best_threshold).astype(int)
    opt_acc = np.mean(opt_preds == true_labels) * 100
    opt_sens = sensitivities[best_idx] * 100
    opt_spec = specificities[best_idx] * 100

    print(f"🏆 Ngưỡng tối ưu (Youden's Threshold) : {best_threshold:.2f}")
    print(f"🔺 Chỉ số Youden cực đại (Max J)      : {best_j:.4f}")
    print(f"🔹 Độ nhạy tối ưu (Sensitivity/Recall) : {opt_sens:.2f}%")
    print(f"🔹 Độ đặc hiệu tối ưu (Specificity)    : {opt_spec:.2f}%")
    print(f"⭐ Độ chính xác tối ưu (Accuracy)      : {opt_acc:.2f}%")
    print("="*70 + "\n")

    return best_threshold, thresholds, sensitivities, specificities, youdens_indices

# =====================================================================
# 4. VISUALIZATION: PLOT ROC CURVE
# =====================================================================
def plot_roc_curve(thresholds, sensitivities, specificities, best_threshold):
    # Tỷ lệ cảnh báo giả False Positive Rate (FPR) = 1 - Specificity [cite: 1361]
    fpr = 1.0 - specificities
    tpr = sensitivities  # True Positive Rate chính là Sensitivity [cite: 1357]

    # Tính toán diện tích dưới đường cong ROC (AUC) sơ bộ bằng phương pháp hình thang
    auc_score = np.trapz(tpr[::-1], fpr[::-1])

    plt.figure(figsize=(7, 6))
    
    # Vẽ đường cong ROC của mô hình
    plt.plot(fpr, tpr, color='crimson', lw=2.5, label=f'P/R Pipeline ROC Curve (AUC = {auc_score:.3f})')
    
    # Vẽ đường phân định ngẫu nhiên (Random guess)
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    
    # Tìm tọa độ của điểm tối ưu trên đồ thị để đánh dấu
    opt_idx = np.where(thresholds == best_threshold)[0][0]
    plt.plot(fpr[opt_idx], tpr[opt_idx], marker='o', color='black', markersize=8, 
         label=f'Optimal Cut-off ({best_threshold:.2f})')

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=13, fontweight='bold')
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.show()

# =====================================================================
# MAIN PIPELINE
# =====================================================================
if __name__ == "__main__":
    set_seed(42)
    
    # Khai báo duy nhất tập dữ liệu test cần đánh giá (Ví dụ: MIMIC AF Cross-Dataset) [cite: 1216, 1256]
    MIMIC_AF_PATH = "/home/linhhima/PPG_ECG/AF_Detection/total_mimic_af_flow_subject.npz"
    
    # Bước 1: Trích xuất đặc trưng trên 1 tập dữ liệu duy nhất [cite: 1257]
    p_ratios, true_labels = load_and_extract_metrics(
        dataset_path=MIMIC_AF_PATH, 
        dataset_name="MIMIC AF Evaluation Dataset",
        sample_ratio=1.0,  # Chạy trên 100% dữ liệu của tập này [cite: 1262]
        fs=125
    )
    
    # Bước 2: Tìm ngưỡng tối ưu thông qua Youden Index chạy từ 0 đến 1 [cite: 1364, 1367]
    best_thresh, thresh_list, sens_list, spec_list, youden_list = optimize_via_youden_index(p_ratios, true_labels)
    
    # Bước 3: Vẽ biểu đồ ROC Curve chèn slide báo cáo 
    plot_roc_curve(thresh_list, sens_list, spec_list, best_thresh)