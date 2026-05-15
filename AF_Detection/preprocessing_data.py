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
from p_wave_detection import get_p_wave_features

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

def pan_tompkins_qrs(ecg_signal: np.ndarray, fs: int = 125):
    """Thuật toán Pan-Tompkins thay thế cho wfdb.xqrs_detect"""
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

def detect_PPG_beats(ppg_signal):
    fs = 125 
    peaks, _ = find_peaks(ppg_signal, height=0.3, distance=int(0.5 * fs))
    rr_intervals = np.diff(peaks) / fs
    return rr_intervals

def detect_ECG_beats(ecg_signal):
    fs = 125
    r_indices = pan_tompkins_qrs(ecg_signal, fs=fs)
    rr_intervals = np.diff(r_indices) / fs
    return rr_intervals

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
        "entropy": [],
        "ratio_peaks": [], 
        "ratio_energy": []  
    }

    for index in tqdm(range(len(ecgs))):
        signal = ecgs[index]
        
        rr_intervals = detect_ECG_beats(signal)
        
        p_ratio, energy_ratio = get_p_wave_features(signal, fs=125)
        
        if rr_intervals is None or len(rr_intervals) < 3:
            features["tpr"].append(0.0)
            features["rmssd"].append(0.0)
            features["entropy"].append(0.0)
            features["ratio_peaks"].append(0.0)
            features["ratio_energy"].append(0.0)
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
        features["ratio_energy"].append(energy_ratio)
    
    tpr_ratio = np.array(features["tpr"])
    rmssd = np.array(features["rmssd"])
    se = np.array(features["entropy"])
    r_peaks = np.array(features["ratio_peaks"])
    r_energy = np.array(features["ratio_energy"])
    
    X_criterion = np.vstack((tpr_ratio, rmssd, se, r_peaks, r_energy)).T
    y = labels
    
    scaler = StandardScaler()
    X_criterion = scaler.fit_transform(X_criterion)   
    
    np.savez(output_name, X=X_criterion, y=y)
    print(f"--- Complete! Save the results at: {output_name} ---")

if __name__ == "__main__":
    set_seed(42)
    load_data_and_extract_features(
        datapath="/home/linhhima/PPG_ECG/AF_Detection/total_mimic_af_recon.npz",
        output_name="/home/linhhima/PPG_ECG/AF_Detection/total_mimic_af_z_score_recon.npz"
    )