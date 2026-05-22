import os
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
from turningPointRatio import turningPointRatio
from rootMeanSquareSuccessiveDifferences import rootMeanSquareSuccessiveDifferences
from shannonEntropy import shannonEntropy
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import messagebox
import random
import torch
from p_wave_detection import get_p_ratio, pan_tompkins_qrs

fs = 125

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

# =====================================================================
# CẬP NHẬT: Thêm tham số external_threshold và return_threshold
# =====================================================================


def detect_PPG_beats(ppg_signal):
    peaks, _ = find_peaks(ppg_signal, height=0.3, distance=int(0.5 * fs))
    rr_intervals = np.diff(peaks) / fs
    return rr_intervals

# =====================================================================
# CẬP NHẬT: Thêm tham số gt_threshold để truyền xuống Pan-Tompkins
# =====================================================================
def detect_ECG_beats(ecg_signal, gt_threshold=None):
    r_indices = pan_tompkins_qrs(ecg_signal, external_threshold=gt_threshold)
    rr_intervals = np.diff(r_indices) / fs
    return rr_intervals

# =====================================================================
# CẬP NHẬT: Thêm tham số dataset_origin vào signature
# =====================================================================
def load_data_and_extract_features(dataset_origin, datapath, output_name):
    
    if not os.path.exists(datapath):
        print(f"Error: Can not find a file in {datapath}")
        return
    if not os.path.exists(dataset_origin):
        print(f"Error: Can not find a file in {dataset_origin}")
        return

    print(f"--- Loading Ground Truth files from {dataset_origin} ---")
    data_origin = np.load(dataset_origin, allow_pickle=True)
    ecgs_origin = data_origin["ecgs"]

    print(f"--- Loading Reconstructed files from {datapath} ---")
    data = np.load(datapath, allow_pickle=True)
    ecgs = data["ecgs"]
    labels = data["labels"]
    
    features = {
        "tpr": [],
        "rmssd": [],
        "entropy": [],
        "ratio_peaks": [], 
    }

    for index in tqdm(range(len(ecgs))):
        signal = ecgs[index]
        gt_signal = ecgs_origin[index]
        
        # BƯỚC MỚI: Tính Threshold chuẩn từ Ground Truth
        _, gt_threshold = pan_tompkins_qrs(gt_signal, return_threshold=True)
        
        # BƯỚC MỚI: Truyền Threshold chuẩn vào hàm lấy RR-Interval
        rr_intervals = detect_ECG_beats(signal, gt_threshold=gt_threshold)
        
        # BƯỚC MỚI: Nhớ sửa hàm get_p_ratio trong file p_wave_detection.py để nhận gt_threshold nhé!
        p_ratio = get_p_ratio(signal, fs=fs, gt_threshold=gt_threshold)
        
        if rr_intervals is None or len(rr_intervals) < 3:
            features["tpr"].append(0.0)
            features["rmssd"].append(0.0)
            features["entropy"].append(0.0)
            features["ratio_peaks"].append(0.0)
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
        features["ratio_peaks"].append(p_ratio)
    
    tpr_ratio = np.array(features["tpr"])
    rmssd = np.array(features["rmssd"])
    se = np.array(features["entropy"])
    r_peaks = np.array(features["ratio_peaks"])
    
    y = labels

    # =====================================================================
    # FIX: CHỈ CHUẨN HÓA 3 FEATURE RR, GIỮ NGUYÊN P_RATIO (0 -> 1)
    # =====================================================================
    # 1. Gom 3 feature cần chuẩn hóa
    X_to_scale = np.vstack((tpr_ratio, rmssd, se)).T
    
    # 2. Áp dụng StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_to_scale)   
    
    # 3. Ghép lại với p_ratio ở dạng thô (nằm ở cột cuối cùng)
    X_criterion = np.hstack((X_scaled, r_peaks.reshape(-1, 1)))
    # =====================================================================
    
    np.savez(output_name, X=X_criterion, y=y)
    print(f"--- Complete! Save the results at: {output_name} ---")

if __name__ == "__main__":
    set_seed(42)
    # Đã sửa lại lỗi cú pháp khi gọi hàm
    load_data_and_extract_features(
        dataset_origin="/home/linhhima/PPG_ECG/datasets/z_score_norm/total_mimic_af.npz",
        datapath="/home/linhhima/PPG_ECG/AF_Detection/total_mimic_af_flow_subject.npz",
        output_name="/home/linhhima/PPG_ECG/AF_Detection/total_mimic_af_flow_subject_1.npz"
    )