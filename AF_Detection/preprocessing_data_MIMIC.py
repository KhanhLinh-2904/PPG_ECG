import os
import wfdb
from wfdb.processing import xqrs_detect
import numpy as np
from scipy.signal import find_peaks
from turningPointRatio import turningPointRatio
from rootMeanSquareSuccessiveDifferences import rootMeanSquareSuccessiveDifferences
from shannonEntropy import shannonEntropy
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import messagebox
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

root = tk.Tk()
root.withdraw()

def detect_PPG_beats(ppg_signal):
    fs = 125 
    # ppg_filtered = bandpass_filter(ppg_signal, fs=fs)
    peaks, _ = find_peaks(ppg_signal, height=0.3, distance=int(0.5 * fs))
    rr_intervals = np.diff(peaks) / fs
    return rr_intervals

def detect_ECG_beats(ecg_signal):
    fs = 125
    qrs_indices = xqrs_detect(sig=ecg_signal, fs=fs)
    # Find R-peaks (maximum value between two QRS detections)
    max_indices = []

    for i in range(len(qrs_indices) - 1):
        start = qrs_indices[i]
        end = qrs_indices[i + 1]

        if end >= len(ecg_signal):
            break

        segment = ecg_signal[start:end+1]
        if len(segment) == 0:
            continue

        max_val = np.max(segment)
        for j in range(start, end + 1):
            if ecg_signal[j] == max_val:
                max_indices.append(j)
                break

    # Extract R-peak amplitudes
    r_peaks = np.array([ecg_signal[idx] for idx in max_indices])

    # Compute RR intervals
    rr_intervals = np.diff(max_indices)/fs
    return rr_intervals

def load_data_and_extract_features(datapath="datasets/total_train.npz", output_name="detect_af_MIMIC_train.npz"):
    """
    Đọc dữ liệu ECG, trích xuất các đặc trưng HRV (TPR, Entropy, RMSSD) và lưu lại.
    """
    if not os.path.exists(datapath):
        print(f"Lỗi: Không tìm thấy file tại {datapath}")
        return

    # 1. Load dữ liệu
    print(f"--- Đang tải dữ liệu từ {datapath} ---")
    data = np.load(datapath, allow_pickle=True)
    ecgs = data["ecgs"]
    labels = data["labels"]
    
    # Khởi tạo các danh sách chứa đặc trưng
    features = {
        "tpr": [],
        "rmssd": [],
        "entropy": []
    }

    print("--- Đang trích xuất đặc trưng ---")
    # 2. Duyệt qua từng tín hiệu ECG (sử dụng tqdm để xem tiến độ)
    for index in tqdm(range(len(ecgs))):
        signal = ecgs[index]
        
        # Giả định hàm detect_ECG_beats trả về mảng các khoảng RR (RR intervals)
        rr_intervals = detect_ECG_beats(signal)
        
        # Kiểm tra điều kiện đủ để tính toán (cần ít nhất 3 nhịp để tính TPR/RMSSD)
        if rr_intervals is None or len(rr_intervals) < 3:
            # Nếu tín hiệu lỗi, gán giá trị 0 hoặc np.nan để tránh crash
            features["tpr"].append(0.0)
            features["rmssd"].append(0.0)
            features["entropy"].append(0.0)
            messagebox.showerror("Error", f"Tín hiệu tại index {index} không đủ dữ liệu RR để tính toán đặc trưng.")
            continue

        # A. Tính Turning Point Ratio (TPR)
        # tpr_actual thường là số điểm đổi chiều tìm được
        _, tpr_actual, _, _ = turningPointRatio(rr_intervals)
        tpr_val = tpr_actual / (len(rr_intervals) - 2)
        
        # B. Tính Shannon Entropy (SE)
        se_val = shannonEntropy(rr_intervals)
        
        # C. Tính RMSSD (Normalized by Mean RR)
        # Công thức: RMSSD / Mean(RR)
        mean_rr = np.mean(rr_intervals)
        if mean_rr > 0:
            rmssd_val = rootMeanSquareSuccessiveDifferences(rr_intervals) / mean_rr
        else:
            rmssd_val = 0.0

        # Thêm kết quả vào danh sách
        features["tpr"].append(tpr_val)
        features["entropy"].append(se_val)
        features["rmssd"].append(rmssd_val)
    
    
    tpr_ratio=np.array(features["tpr"])
    rmssd=np.array(features["rmssd"])
    se=np.array(features["entropy"])
    # 3. Lưu kết quả
    
    X_criterion = np.vstack((tpr_ratio, rmssd, se)).T
    y = labels
    scaler = StandardScaler()           # Transform val using train scaler
    X_criterion = scaler.fit_transform(X_criterion)   
    np.savez(output_name, X=X_criterion, y=y)
    print(f"--- Hoàn tất! Đã lưu kết quả tại: {output_name} ---")

if __name__ == "__main__":
    set_seed(42)
    load_data_and_extract_features(
        datapath="AF_Detection/total_test_ecg_reconstructions_no_align.npz",
        output_name="AF_Detection/detect_af_MIMIC_test_recon_no_align.npz"
    )
    
    