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

root = tk.Tk()
root.withdraw()

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
    total_power_1_3 = 0.0
    total_power_3_10 = 0.0
    
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
                abs_p_peak = start_idx + peak_idx_local
                p_peaks_indices.append(abs_p_peak)
                
            windowed = p_detrended * np.hamming(len(p_detrended))
            N = len(windowed)
            freqs = fftfreq(N, 1/fs)[:N//2]
            energy = np.abs(fft(windowed))[:N//2] ** 2
            
            power_1_3Hz = np.sum(energy[(freqs >= 1.0) & (freqs < 3.0)])
            power_3_10Hz = np.sum(energy[(freqs >= 3.0)])
            
            total_power_1_3 += power_1_3Hz
            total_power_3_10 += power_3_10Hz

    valid_beats = len(r_indices) - 1
    p_ratio = num_p_found / valid_beats if valid_beats > 0 else 0
    
    if total_power_1_3 == 0: 
        total_power_1_3 = 1e-6 
        
    avg_energy_ratio = (total_power_3_10 / total_power_1_3) if valid_beats > 0 else 0.0
    
    return p_ratio, avg_energy_ratio, p_peaks_indices, r_indices

def run_pure_p_wave_classification(dataset_path, dataset_origin_path, num_plots_to_show=100, filter_view="all"):
    print(f"[*] Loading Reconstructed data from {dataset_path}...")
    if not os.path.exists(dataset_path):
        print("[!] dataset_path file not found.")
        return
    data = np.load(dataset_path, allow_pickle=True)
    ecgs = data["ecgs"]
    labels = data["labels"]
    raw_records = data["records"]
    record_name = raw_records
    
    data_orig = np.load(dataset_origin_path, allow_pickle=True)
    ecgs_orig = data_orig["ecgs"]
    
    correct_af, correct_non_af = 0, 0
    total_af, total_non_af = 0, 0
    
    P_RATIO_THRESH = 0.44

    print("\n[*] Analyzing P-Waves with Ground Truth Threshold...")
    
    plots_shown = 0 

    for i in range(len(ecgs)):
        sig = ecgs[i]
        true_sig = ecgs_orig[i]
        true_lbl = labels[i]
        name_rec = record_name[i]
        
        _, gt_threshold = pan_tompkins_qrs(true_sig, fs=125, return_threshold=True)

        p_ratio, energy_ratio, p_peaks_idx, r_idx = analyze_p_wave_for_af(sig, fs=125, gt_threshold=gt_threshold)
        
        if p_ratio > P_RATIO_THRESH:
            pred_lbl = 0 
        else:
            pred_lbl = 1 
            
        if true_lbl == 1:
            total_af += 1
            if pred_lbl == 1: correct_af += 1
        else:
            total_non_af += 1
            if pred_lbl == 0: correct_non_af += 1

        show_this_plot = False
        if filter_view == "all":
            show_this_plot = True
        elif filter_view == "af" and true_lbl == 1:
            show_this_plot = True
        elif filter_view == "non_af" and true_lbl == 0:
            show_this_plot = True

        if show_this_plot and plots_shown < num_plots_to_show:
            fig, ax = plt.subplots(1, 1, figsize=(12, 4))
            time_ax = np.arange(len(sig)) / 125
            
            title_color = 'darkgreen' if true_lbl == pred_lbl else 'darkred'
            true_str = "AF" if true_lbl == 1 else "Non-AF"
            pred_str = "AF" if pred_lbl == 1 else "Non-AF"
            
            ax.plot(time_ax, sig, color='black', linewidth=1, alpha=0.8, label='Reconstructed ECG')
            
            if len(r_idx) > 0:
                ax.scatter(r_idx/125, sig[r_idx], color='red', marker='v', s=60, zorder=3, label='R-Peak')
            
            if len(p_peaks_idx) > 0:
                p_peaks_arr = np.array(p_peaks_idx)
                ax.scatter(p_peaks_arr/125, sig[p_peaks_arr], color='limegreen', marker='o', s=50, zorder=4, label='P-Peak')

            if len(r_idx) > 1:
                for j in range(1, len(r_idx)):
                    r_curr = r_idx[j]
                    r_prev = r_idx[j-1]
                    rr_dist = r_curr - r_prev
                    
                    s_time = max(0, (r_curr - int(rr_dist * 0.45)) / 125)
                    e_time = max(0, (r_curr - int(0.05 * rr_dist)) / 125)
                    
                    ax.axvspan(s_time, e_time, color='gray', alpha=0.15)

            ax.set_title(f"Record {name_rec} | Ground Truth: {true_str} | Predicted: {pred_str}\nP-over-R-Ratio: {p_ratio:.2f}", fontweight='bold', color=title_color)
            ax.set_xlabel("Time (sec)")
            ax.set_ylabel("Amplitude")
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show(block=True)
            
            plots_shown += 1

    tp = correct_af
    tn = correct_non_af
    fp = total_non_af - correct_non_af
    fn = total_af - correct_af
    total_records = total_af + total_non_af

    accuracy = ((tp + tn) / total_records * 100) if total_records > 0 else 0
    recall = (tp / total_af * 100) if total_af > 0 else 0 
    specificity = (tn / total_non_af * 100) if total_non_af > 0 else 0
    precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0

    print("\n" + "="*65)
    print(" 📊 ATRIAL FIBRILLATION DIAGNOSIS SUMMARY (P-WAVE ONLY)")
    print("="*65)
    print(f"🔹 Total test records     : {total_records} records")
    print("-" * 65)
    print(f"🔴 ATRIAL FIBRILLATION (AF) - POSITIVE CLASS:")
    print(f"   - Actual cases (TP+FN)  : {total_af}")
    print(f"   - True Positives (TP)   : {tp}")
    print(f"   - False Negatives (FN)  : {fn} (Missed AF)")
    print("-" * 65)
    print(f"🟢 NORMAL (Non-AF) - NEGATIVE CLASS:")
    print(f"   - Actual cases (TN+FP)  : {total_non_af}")
    print(f"   - True Negatives (TN)   : {tn}")
    print(f"   - False Positives (FP)  : {fp} (False Alarms)")
    print("="*65)
    print(" 🏆 FINAL EVALUATION METRICS:")
    print(f"   ➤ Accuracy              : {accuracy:.2f}%")
    print(f"   ➤ Precision             : {precision:.2f}%")
    print(f"   ➤ Recall (Sensitivity)  : {recall:.2f}%")
    print(f"   ➤ Specificity           : {specificity:.2f}%")
    print("="*65 + "\n")

if __name__ == "__main__":
    set_seed(42)
    dataset_path = "/home/linhhima/PPG_ECG/AF_Detection/total_mimic_af_flow_segment.npz"
    dataset_origin = "/home/linhhima/PPG_ECG/datasets/z_score_norm/total_mimic_af.npz"
    
    run_pure_p_wave_classification(
        dataset_path=dataset_path, 
        dataset_origin_path=dataset_origin, 
        num_plots_to_show=100, 
        filter_view="af"
    )