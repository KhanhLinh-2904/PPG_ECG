import os
import wfdb
from wfdb.processing import xqrs_detect
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fftpack import fft, fftfreq
from scipy.signal.windows import hamming
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import tkinter as tk
import random
import torch
from scipy.signal import butter, filtfilt, find_peaks
# Import các hàm trích xuất đặc trưng của bạn
from turningPointRatio import turningPointRatio
from rootMeanSquareSuccessiveDifferences import rootMeanSquareSuccessiveDifferences
from shannonEntropy import shannonEntropy

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
# THUẬT TOÁN PAN-TOMPKINS
# =====================================================================
def pan_tompkins_qrs(ecg_signal: np.ndarray, fs: int = 125, external_threshold: float = None, return_threshold: bool = False):
    """
    Thuật toán Pan-Tompkins đã chỉnh sửa để nhận/trả ngưỡng.
    """
    ecg_signal = np.array(ecg_signal).flatten()
    
    # 1. Bandpass Filter (Lọc thông dải 5-15 Hz)
    nyq = 0.5 * fs
    low = 5.0 / nyq
    high = 15.0 / nyq
    b, a = butter(1, [low, high], btype='band')
    filtered_ecg = filtfilt(b, a, ecg_signal)
    
    # 2. Derivative (Đạo hàm)
    diff_ecg = np.diff(filtered_ecg)
    diff_ecg = np.insert(diff_ecg, 0, diff_ecg[0])
    
    # 3. Squaring (Bình phương)
    squared_ecg = diff_ecg ** 2
    
    # 4. Moving Window Integration (Tích phân cửa sổ trượt)
    window_width = int(0.15 * fs)
    integrated_ecg = np.convolve(squared_ecg, np.ones(window_width) / window_width, mode='same')
    
    # 5. THRESHOLDING (Ngưỡng thích nghi)
    # Nếu có truyền ngưỡng từ ngoài vào thì dùng nó, nếu không thì tự tính trung bình
    if external_threshold is not None:
        threshold = external_threshold
    else:
        threshold = np.mean(integrated_ecg)
        
    min_distance = int(0.3 * fs)
    peaks_integrated, _ = find_peaks(integrated_ecg, height=threshold, distance=min_distance)
    
    # 6. Back-search (Tìm ngược lại đỉnh R chính xác trên tín hiệu gốc)
    r_peaks = []
    search_window = int(0.05 * fs)
    
    for p in peaks_integrated:
        start = max(0, p - search_window)
        end = min(len(ecg_signal), p + search_window)
        if start < end:
            local_max = np.argmax(ecg_signal[start:end])
            r_peaks.append(start + local_max)
            
    # Trả về cả đỉnh R và cái ngưỡng đã dùng (nếu được yêu cầu)
    if return_threshold:
        return np.array(r_peaks), threshold
    
    return np.array(r_peaks)

# =====================================================================
# HÀM DETECT BEATS (ĐÃ TỐI ƯU HÓA)
# =====================================================================
def detect_ECG_beats_with_indices(ecg_signal):
    """
    Trợ thủ tìm đỉnh R và trả về cả khoảng RR lẫn mảng vị trí đỉnh R.
    Sử dụng thuật toán Pan-Tompkins tự xây dựng.
    """
    fs = 125
    
    # Chỉ cần gọi trực tiếp Pan-Tompkins, mảng trả về đã là đỉnh R chuẩn xác
    r_indices = pan_tompkins_qrs(ecg_signal, fs=fs)
    
    # Tránh lỗi nếu tín hiệu quá ngắn hoặc nhiễu không tìm được đủ 2 đỉnh R
    if len(r_indices) < 2:
        return np.array([]), np.array([])
        
    # Tính toán khoảng R-R (đổi ra đơn vị giây)
    rr_intervals = np.diff(r_indices) / fs
    
    return rr_intervals, r_indices

# =====================================================================
# KỸ THUẬT TRIỆT TIÊU TÂM THẤT (VAC / TEMPLATE SUBTRACTION)
# =====================================================================
def cancel_ventricular_activity(ecg_signal, r_indices, fs=125):
    """
    Tính khuôn mẫu (Template) chứa QRS & T, sau đó trừ khỏi tín hiệu gốc.
    """
    win_before = int(0.05 * fs) 
    win_after = int(0.40 * fs)  
    
    valid_segments = []
    valid_r = []
    
    for r in r_indices:
        start = r - win_before
        end = r + win_after
        if start >= 0 and end < len(ecg_signal):
            valid_segments.append(ecg_signal[start:end])
            valid_r.append(r)
            
    if len(valid_segments) == 0:
        return ecg_signal

    template = np.median(valid_segments, axis=0)
    
    taper_len = int(0.02 * fs) # Taper 20ms
    if taper_len > 0:
        taper = np.ones(len(template))
        taper[:taper_len] = np.linspace(0, 1, taper_len)
        taper[-taper_len:] = np.linspace(1, 0, taper_len)
        template = template * taper

    residual_signal = np.copy(ecg_signal)
    for r in valid_r:
        start = r - win_before
        end = r + win_after
        residual_signal[start:end] -= template
        
    return residual_signal

def compute_residual_spectrum(residual_signal, fs=125):
    """
    Tính phổ năng lượng FFT của tín hiệu sau khi đã bị triệt tiêu Tâm thất.
    """
    res_detrend = residual_signal - np.mean(residual_signal)
    windowed = res_detrend * hamming(len(res_detrend))
    
    N = len(windowed)
    freqs = fftfreq(N, 1/fs)[:N//2]
    spec = np.abs(fft(windowed))[:N//2]
    
    if np.max(spec) > 0:
        spec = spec / np.max(spec)
        
    return freqs, spec

# =====================================================================
# HÀM DỰ ĐOÁN AF DỰA VÀO LUẬT HEURISTIC (BIÊN ĐỘ > 0.4 TẠI >= 4 HZ)
# =====================================================================
def predict_af_from_spectrum(freqs, spec, freq_threshold=4.0, amp_threshold=0.4):
    """
    Luật dự đoán cứng: Nếu tồn tại tần số >= 4Hz có biên độ > 0.4 -> AF (Trả về 1).
    Nếu không -> Non-AF (Trả về 0).
    """
    if freqs is None or spec is None:
        return 0
        
    valid_freq_indices = np.where(freqs >= freq_threshold)[0]
    
    if len(valid_freq_indices) == 0:
        return 0
        
    valid_amps = spec[valid_freq_indices]
    
    if np.any(valid_amps > amp_threshold):
        return 1
    else:
        return 0

# =====================================================================
# HÀM VẼ ĐỒ THỊ KIỂM CHỨNG
# =====================================================================
def visualize_single_record(signal, r_indices, label, record_index, fs=125):
    residual_signal = cancel_ventricular_activity(signal, r_indices, fs)
    freqs, spec = compute_residual_spectrum(residual_signal, fs)
    
    prediction = predict_af_from_spectrum(freqs, spec, freq_threshold=4.0, amp_threshold=0.4)
    pred_str = "AF" if prediction == 1 else "Non-AF"
    label_str = "AF" if label == 1 else "Non-AF"
    
    match_str = "ĐÚNG" if prediction == label else "SAI"
    title_color = "darkgreen" if prediction == label else "darkred"
    
    fig, axs = plt.subplots(3, 1, figsize=(12, 9))
    time_ax = np.arange(len(signal)) / fs
    
    axs[0].plot(time_ax, signal, color='black', linewidth=1)
    axs[0].scatter(r_indices/fs, signal[r_indices], color='red', marker='v', zorder=3, label='R-peaks')
    axs[0].set_title(f"Record {record_index} | Thực tế: [{label_str}] - 1. Tín hiệu Gốc (Có Sóng T)", fontweight='bold')
    axs[0].set_ylabel("Amplitude")
    axs[0].legend(loc="upper right")
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(time_ax, residual_signal, color='blue', linewidth=1)
    axs[1].set_title("2. Tín hiệu Dư (Sau khi Triệt tiêu QRS và Sóng T)", fontweight='bold')
    axs[1].set_ylabel("Amplitude")
    for r in r_indices:
        axs[1].axvline(x=r/fs, color='red', alpha=0.1, linestyle='--')
    axs[1].grid(True, alpha=0.3)

    axs[2].plot(freqs, spec, color='purple', linewidth=1.5)
    
    # Cập nhật hiển thị text theo mức 0.4
    title_str = (f"3. Phổ Năng Lượng Tâm Nhĩ | "
                 f"Máy dự đoán: [{pred_str}] -> {match_str} "
                 f"(Dựa trên luật: Tần số >= 4Hz có biên độ > 0.4)")
    axs[2].set_title(title_str, fontweight='bold', color=title_color)
    
    axs[2].set_xlabel("Frequency (Hz)")
    axs[2].set_ylabel("Normalized Amplitude")
    axs[2].set_xlim(0, 15)
    
    # Vẽ đường đứt nét làm mốc Ngưỡng Tần số (4 Hz) và Ngưỡng Biên độ (0.4)
    axs[2].axvline(x=4.0, color='gray', linestyle='--', alpha=0.8, label='Ngưỡng Tần số (4Hz)')
    axs[2].axhline(y=0.4, color='orange', linestyle='--', alpha=0.8, label='Ngưỡng Biên độ (0.4)')
    
    axs[2].axvspan(4.0, 15.0, color='red', alpha=0.05)
    axs[2].legend(loc="upper right")
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show(block=True)

# =====================================================================
# HÀM TRÍCH XUẤT ĐẶC TRƯNG CHÍNH
# =====================================================================
def load_data_and_extract_features(datapath="datasets/total_train.npz", output_name="detect_af_MIMIC_train.npz"):
    if not os.path.exists(datapath):
        print(f"Error: Can not find a file in {datapath}")
        return

    print(f"--- Loading files {datapath} ---")
    data = np.load(datapath, allow_pickle=True)
    ecgs = data["ecgs"]
    labels = data["labels"]
    
    features = {
        "tpr": [],
        "rmssd": [],
        "entropy": []
    }

    for index in tqdm(range(len(ecgs)), desc="Extracting Features"):
        signal = ecgs[index]
        
        rr_intervals, r_indices = detect_ECG_beats_with_indices(signal)
        
        if len(rr_intervals) < 3:
            features["tpr"].append(0.0)
            features["rmssd"].append(0.0)
            features["entropy"].append(0.0)
            continue

        _, tpr_actual, _, _ = turningPointRatio(rr_intervals)
        tpr_val = tpr_actual / (len(rr_intervals) - 2)
        
        se_val = shannonEntropy(rr_intervals)
        
        mean_rr = np.mean(rr_intervals)
        if mean_rr > 0:
            rmssd_val = rootMeanSquareSuccessiveDifferences(rr_intervals) / mean_rr
        else:
            rmssd_val = 0.0

        features["tpr"].append(tpr_val)
        features["entropy"].append(se_val)
        features["rmssd"].append(rmssd_val)
    
    tpr_ratio = np.array(features["tpr"])
    rmssd = np.array(features["rmssd"])
    se = np.array(features["entropy"])
    
    X_criterion = np.vstack((tpr_ratio, rmssd, se)).T
    y = labels
    scaler = StandardScaler()
    X_criterion = scaler.fit_transform(X_criterion)   
    np.savez(output_name, X=X_criterion, y=y)
    print(f"--- Complete! Save the results at: {output_name} ---")

if __name__ == "__main__":
    set_seed(42)
    
    dataset_path = "/home/linhhima/PPG_ECG/AF_Detection/total_mimic_af_recon.npz"
    print(f"[*] Đang load dữ liệu từ {dataset_path} để kiểm chứng...")
    if os.path.exists(dataset_path):
        demo_data = np.load(dataset_path, allow_pickle=True)
        demo_ecgs = demo_data["ecgs"]
        demo_labels = demo_data["labels"]
        
        print("\n" + "="*50)
        print("HƯỚNG DẪN KIỂM CHỨNG:")
        print("- Cửa sổ hiển thị 3 đồ thị (Gốc, Sau khi triệt tiêu QRS/T, và Phổ Tần Số).")
        # Đã cập nhật text ở terminal để hiển thị đúng 0.4
        print("- Đồ thị 3 sẽ tự động dự đoán AF dựa trên luật Tần số >= 4Hz có biên độ > 0.4.")
        print("- Ấn dấu [X] để xem bản ghi tiếp theo.")
        print("- Bấm Ctrl + C trên terminal để thoát.")
        print("="*50 + "\n")
        
        for i in range(len(demo_ecgs)):
            sig = demo_ecgs[i]
            lbl = demo_labels[i]
            if lbl == 0:  
                _, r_idx = detect_ECG_beats_with_indices(sig)
                visualize_single_record(sig, r_idx, lbl, record_index=i, fs=125)
            
    else:
        print(f"[!] Không tìm thấy file {dataset_path}.")