import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from load_data import LoadData
from CLIP import ECG_PPG_Fusion_Model  
import random
import os
from scipy.signal import butter, filtfilt, find_peaks
from metric import calculate_cosine_similarity, calculate_dtw_distance, calculate_metrics
from scipy.signal import correlate
from tqdm import tqdm

# --- CONFIGURATION ---
SEED = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
INPUT_LENGTH = 2400
OUTPUT_EMBED_DIM = 128
SEQ_LENGTH = 2400
TEST_DATA_PATH = '/home/linhhima/Diffusion_datasets/combined_segment_split_test.npz'
CLIP_MODEL_PATH = '/home/linhhima/PPG_ECG/PPG2ECG_CLIP/best_multitask_model_segment.pth'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def pan_tompkins_qrs(ecg_signal: np.ndarray, fs: int = 125, external_threshold: float = None, return_threshold: bool = False):
   
    ecg_signal = np.array(ecg_signal).flatten()
    
    nyq = 0.5 * fs
    low = 5.0 / nyq
    high = 15.0 / nyq
    b, a = butter(1, [low, high], btype='band')
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

def calculate_peak_count_ratio(true_s: np.ndarray, pred_s: np.ndarray, sampling_rate=125):
   
    true_peaks, gt_threshold = pan_tompkins_qrs(true_s, fs=sampling_rate, return_threshold=True)
    
    pred_peaks = pan_tompkins_qrs(pred_s, fs=sampling_rate, external_threshold=gt_threshold, return_threshold=False)
    
    num_true = len(true_peaks)
    num_pred = len(pred_peaks)
    
    if num_true > 0:
        ratio = num_pred / num_true
    else:
        ratio = 1.0 if num_pred == 0 else 0.0
        
    return num_true, num_pred, ratio

def load_model():
    print(f"Loading model on {DEVICE}...")
    
    model = ECG_PPG_Fusion_Model(embed_dim=OUTPUT_EMBED_DIM, freq_dim=256, target_length=SEQ_LENGTH).to(DEVICE)

    if torch.cuda.is_available():
        weights = torch.load(CLIP_MODEL_PATH)
    else:
        weights = torch.load(CLIP_MODEL_PATH, map_location='cpu')

    model.load_state_dict(weights)
    model.eval()
    
    return model

def save_ecg_reconstruction(output_path="AF_Detection/ecg_reconstructions.npz"):
    set_seed(SEED)
    model = load_model()
    
    all_predicted_ecgs = []
    all_original_ppgs = []
    all_labels = []
    all_record_names = []

    try:
        test_dataset = LoadData(TEST_DATA_PATH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    except Exception as e:
        print(f"Error: {e}")
        return
   
    print("Running inference and saving reconstructions...")
    with torch.no_grad():
        for i, (ecg, ppg, record_names, label) in enumerate(test_loader):
            ppg_input = ppg.to(DEVICE).float().unsqueeze(1)
            # ecg_input = ecg.to(DEVICE).float().unsqueeze(1)

            predicted_ecg = model(None,ppg_input)
            
            all_predicted_ecgs.append(predicted_ecg.squeeze(1).cpu().numpy())
            all_original_ppgs.append(ppg.cpu().numpy())
            all_labels.append(label.cpu().numpy())
            all_record_names.append(record_names)
            
    save_dict = {
        "ecgs": np.concatenate(all_predicted_ecgs, axis=0),
        "ppgs": np.concatenate(all_original_ppgs, axis=0),
        "labels": np.concatenate(all_labels, axis=0),
        "records": np.array(all_record_names, dtype=object) 
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, **save_dict)
    print(f"Saved reconstructions to {output_path}")

def get_integrated_energy(signal, fs=125):
    nyq = 0.5 * fs
    b, a = butter(1, [5.0 / nyq, 15.0 / nyq], btype='band')
    filtered = filtfilt(b, a, signal)
    diff = np.diff(filtered)
    diff = np.insert(diff, 0, diff[0])
    squared = diff ** 2
    window = int(0.15 * fs)
    return np.convolve(squared, np.ones(window) / window, mode='same')


def visualize_results(ppg, ecg_true, ecg_pred, record_name, fs=125):
    print(f"Visualizing Record: {record_name}")
    
    true_peaks, gt_thresh = pan_tompkins_qrs(ecg_true, fs=fs, return_threshold=True)
    pred_peaks = pan_tompkins_qrs(ecg_pred, fs=fs, external_threshold=gt_thresh, return_threshold=False)
    
    num_true = len(true_peaks)
    num_pred = len(pred_peaks)
    ratio = (num_pred / num_true * 100) if num_true > 0 else 0.0

    int_true = get_integrated_energy(ecg_true, fs)
    int_pred = get_integrated_energy(ecg_pred, fs)

    t_ppg = np.arange(len(ppg))
    t_ecg = np.arange(len(ecg_true))

    plt.figure(figsize=(14, 15))
    plt.suptitle(
        f"Record: {record_name} | R-Peaks Ratio: {ratio:.1f}%\n"
        f"GT Peaks: {num_true} | Pred Peaks: {num_pred}", 
        fontsize=15, fontweight='bold'
    )

    # --- 1: Input PPG ---
    plt.subplot(5, 1, 1)
    plt.plot(t_ppg, ppg, color='green', label='Input PPG')
    plt.title("Input PPG Signal")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    # ---  2: Ground Truth ECG + R-Peaks ---
    plt.subplot(5, 1, 2)
    plt.plot(t_ecg, ecg_true, color='blue', label='Ground Truth ECG')
    if num_true > 0:
        plt.scatter(true_peaks, ecg_true[true_peaks], color='red', marker='x', s=80, label=f'GT Peaks (n={num_true})', zorder=5)
    plt.title("Ground Truth ECG")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    # ---  3: Predicted ECG + R-Peaks ---
    plt.subplot(5, 1, 3)
    plt.plot(t_ecg, ecg_pred, color='red', label='Predicted ECG')
    if num_pred > 0:
        plt.scatter(pred_peaks, ecg_pred[pred_peaks], color='orange', marker='x', s=80, label=f'Pred Peaks (n={num_pred})', zorder=5)
    plt.title("Predicted ECG (Reconstructed)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    # ---  4: Comparison - Integrated Energy & Threshold ---
    ax_energy = plt.subplot(5, 1, 4)
    
    # Energy of Ground Truth
    ax_energy.plot(t_ecg, int_true, color='gray', alpha=0.6, label='GT Integrated Energy')
    # Energy of Predicted
    ax_energy.plot(t_ecg, int_pred, color='dodgerblue', alpha=0.6, label='Pred Integrated Energy')
    
    #  Threshold
    ax_energy.axhline(y=gt_thresh, color='green', linestyle='-.', linewidth=2, label=f'Threshold ({gt_thresh:.4f})')
    
    ax_energy.text(0.01, 0.85, f'Ratio: {ratio:.1f}%', transform=ax_energy.transAxes, 
             fontsize=12, fontweight='bold', color='darkred',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    ax_energy.set_title("Pan-Tompkins: Integrated Energy & Detection Threshold")
    ax_energy.set_ylabel("Energy")
    ax_energy.grid(True, alpha=0.3)
    ax_energy.legend(loc='upper right')

    # ---  5: Comparison - Raw ECG Overlap ---
    plt.subplot(5, 1, 5)
    plt.plot(t_ecg, ecg_true, color='black', label='Ground Truth', alpha=0.7, linewidth=1.5)
    plt.plot(t_ecg, ecg_pred, color='red', label='Predicted', linestyle='--', alpha=0.8, linewidth=1.5)
    
    if num_true > 0:
        plt.scatter(true_peaks, ecg_true[true_peaks], color='blue', marker='o', s=40, zorder=5)
    if num_pred > 0:
        plt.scatter(pred_peaks, ecg_pred[pred_peaks], color='orange', marker='x', s=60, zorder=6)
        
    plt.title("Comparison: Raw Ground Truth vs Predicted ECG")
    plt.xlabel("Time Samples")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    plt.show() 

def run_visualization():
    set_seed(SEED)
    model = load_model()
    seen_records = set()
    
    target_records = {
        "dalia_S11", 
        "wesad_S14", 
        "bidmc_bidmc08", 
        "capno_0028_8min", 
        "mimic_64"
    }

    try:
        test_dataset = LoadData(TEST_DATA_PATH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    except Exception as e:
        print(f"Error: {e}")
        return
   
    print("[*] Running inference to find and visualize specific target records...")
    with torch.no_grad():
        for i, (ecg, ppg, record_names) in enumerate(test_loader):
            ppg_input = ppg.to(DEVICE).float().unsqueeze(1)
            ecg_input = ecg.to(DEVICE).float().unsqueeze(1)

            _, _, predicted_ecg = model(ecg_input, ppg_input)

            ppg_np = np.atleast_2d(ppg_input.cpu().squeeze().numpy())
            ecg_true_np = np.atleast_2d(ecg.cpu().squeeze().numpy())
            ecg_pred_np = np.atleast_2d(predicted_ecg.cpu().squeeze().numpy())

            for idx in range(ppg_np.shape[0]):
                current_rec_name = record_names[idx]
                
                if current_rec_name in target_records and current_rec_name not in seen_records:
                    visualize_results(
                        ppg_np[idx], 
                        ecg_true_np[idx], 
                        ecg_pred_np[idx], 
                        current_rec_name
                    )
                    seen_records.add(current_rec_name)
                    
                    if len(seen_records) == len(target_records):
                        print("\n[*] Complete!")
                        return


# def visualize_results(ppg, ecg_true, ecg_pred, record_name):
#     print(f"Visualizing Record: {record_name}")
#     t_ppg = np.arange(len(ppg))
#     t_ecg = np.arange(len(ecg_true))

#     plt.figure(figsize=(12, 10))
#     plt.suptitle(f"Record: {record_name}", fontsize=14, fontweight='bold')

#     plt.subplot(4, 1, 1)
#     plt.plot(t_ppg, ppg, color='green', label='Input PPG')
#     plt.title("Input PPG Signal")
#     plt.grid(True, alpha=0.3)
#     plt.legend(loc='upper right')

#     plt.subplot(4, 1, 2)
#     plt.plot(t_ecg, ecg_true, color='blue', label='Ground Truth ECG')
#     plt.title("Ground Truth ECG")
#     plt.grid(True, alpha=0.3)
#     plt.legend(loc='upper right')

#     plt.subplot(4, 1, 3)
#     plt.plot(t_ecg, ecg_pred, color='red', label='Predicted ECG')
#     plt.title("Predicted ECG (Reconstructed)")
#     plt.grid(True, alpha=0.3)
#     plt.legend(loc='upper right')

#     plt.subplot(4, 1, 4)
#     plt.plot(t_ecg, ecg_true, color='black', label='Ground Truth', alpha=0.7)
#     plt.plot(t_ecg, ecg_pred, color='red', label='Predicted', linestyle='--', alpha=0.8)
#     plt.title("Comparison: Ground Truth vs Predicted ECG")
#     plt.xlabel("Time Samples")
#     plt.grid(True, alpha=0.3)
#     plt.legend(loc='upper right')

#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.show()

def align_signals(true_s: np.ndarray, pred_s: np.ndarray) -> np.ndarray:
    
    correlation = correlate(true_s, pred_s, mode='full')
    lag = np.argmax(correlation) - (len(pred_s) - 1)
    
    aligned_pred = np.zeros_like(pred_s)
    
    if lag > 0:
        aligned_pred[lag:] = pred_s[:-lag]
    elif lag < 0:
        aligned_pred[:lag] = pred_s[-lag:]
    else:
        aligned_pred = pred_s.copy()
        
    return aligned_pred

def run_loss():
    model = load_model()
    
    try:
        test_dataset = LoadData(TEST_DATA_PATH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    except Exception as e:
        print(f"Error: {e}")
        return

    target_datasets = ["dalia", "wesad", "bidmc", "capno", "mimic"]
    
    def get_empty_metrics():
        return {
            'before': {'rmse': 0.0, 'pearson': 0.0, 'dtw': 0.0, 'cosine': 0.0},
            'after':  {'rmse': 0.0, 'pearson': 0.0, 'dtw': 0.0, 'cosine': 0.0},
            'count': 0
        }

    all_metrics = {ds: get_empty_metrics() for ds in target_datasets}
    all_metrics["overall"] = get_empty_metrics()

    print("[*] Calculating Metrics per Dataset and Overall...")
    with torch.no_grad():
        for i, (ecg, ppg, record_names) in enumerate(test_loader):
            ppg_input = ppg.to(DEVICE).float().unsqueeze(1)
            ecg_input = ecg.to(DEVICE).float().unsqueeze(1)

            _, _, predicted_ecg = model(ecg_input, ppg_input)

            ecg_true_np = np.atleast_2d(ecg.cpu().squeeze().numpy())
            ecg_pred_np = np.atleast_2d(predicted_ecg.cpu().squeeze().numpy())

            for b in range(ecg_true_np.shape[0]):
                true_s = ecg_true_np[b]
                pred_s = ecg_pred_np[b]
                
                rec_name = str(record_names[b]).lower()

                rmse_b, pearson_b = calculate_metrics(true_s, pred_s)
                dtw_b = calculate_dtw_distance(true_s, pred_s)
                cosine_b = calculate_cosine_similarity(true_s, pred_s)
                
                aligned_pred_s = align_signals(true_s, pred_s)

                rmse_a, pearson_a = calculate_metrics(true_s, aligned_pred_s)
                dtw_a = calculate_dtw_distance(true_s, aligned_pred_s)
                cosine_a = calculate_cosine_similarity(true_s, aligned_pred_s)

                current_ds = None
                for ds in target_datasets:
                    if ds in rec_name:
                        current_ds = ds
                        break
                
                def add_to_metrics(metric_dict):
                    metric_dict['before']['rmse'] += rmse_b
                    metric_dict['before']['pearson'] += pearson_b
                    metric_dict['before']['dtw'] += dtw_b
                    metric_dict['before']['cosine'] += cosine_b
                    
                    metric_dict['after']['rmse'] += rmse_a
                    metric_dict['after']['pearson'] += pearson_a
                    metric_dict['after']['dtw'] += dtw_a
                    metric_dict['after']['cosine'] += cosine_a
                    
                    metric_dict['count'] += 1

                if current_ds is not None:
                    add_to_metrics(all_metrics[current_ds])
                
                add_to_metrics(all_metrics["overall"])

 
    def print_report(title, m_dict):
        count = m_dict['count']
        if count == 0:
            return 
            
        print(f"\n{'='*55}")
        print(f"RESULTS: {title.upper()} ({count} samples)")
        print(f"{'='*55}")
        print(f"{'Metric':<12} | {'Before Align':<15} | {'After Align':<15}")
        print(f"{'-'*55}")
        print(f"RMSE         | {m_dict['before']['rmse']/count:<15.4f} | {m_dict['after']['rmse']/count:<15.4f}")
        print(f"Pearson      | {m_dict['before']['pearson']/count:<15.4f} | {m_dict['after']['pearson']/count:<15.4f}")
        print(f"DTW          | {m_dict['before']['dtw']/count:<15.4f} | {m_dict['after']['dtw']/count:<15.4f}")
        print(f"Cosine       | {m_dict['before']['cosine']/count:<15.4f} | {m_dict['after']['cosine']/count:<15.4f}")
        print(f"{'='*55}")

    for ds in target_datasets:
        print_report(f"DATASET: {ds}", all_metrics[ds])
        
    print_report("OVERALL DATASET", all_metrics["overall"])

def run_peak_count_evaluation():
    set_seed(SEED)
    model = load_model()
    
    try:
        test_dataset = LoadData(TEST_DATA_PATH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) 
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        return

    target_datasets = ["dalia", "wesad", "bidmc", "capno", "mimic"]
    
    def get_empty_tracker():
        return {
            'sum_abs_err': 0.0, 
            'sum_err_pct': 0.0, 
            'count': 0
        }
        
    stats = {ds: get_empty_tracker() for ds in target_datasets}
    stats["overall"] = get_empty_tracker()

    print("[*] Evaluating R-Peaks (MAE & Error Percentage)...")
    print("[!] USING GROUND TRUTH THRESHOLD FOR PREDICTED ECG")
    
    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(test_loader, desc="Evaluating R-Peaks Errors")):
            ecg = batch_data[0]
            ppg = batch_data[1]
            
            record_names = batch_data[2] if len(batch_data) > 2 else [f"unknown_{j}" for j in range(ecg.shape[0])]
            
            ppg_input = ppg.to(DEVICE).float().unsqueeze(1)
            ecg_input = ecg.to(DEVICE).float().unsqueeze(1)

            _, _, predicted_ecg = model(ecg_input, ppg_input)

            ecg_true_np = np.atleast_2d(ecg.cpu().squeeze().numpy())
            ecg_pred_np = np.atleast_2d(predicted_ecg.cpu().squeeze().numpy())

            for b in range(ecg_true_np.shape[0]):
                true_s = ecg_true_np[b]
                pred_s = ecg_pred_np[b]
                rec_name = str(record_names[b]).lower()

                num_true, num_pred, _ = calculate_peak_count_ratio(true_s, pred_s, sampling_rate=125)
                
               
                abs_err = abs(num_true - num_pred)
                err_pct = (abs_err / num_true * 100) if num_true > 0 else 0.0
                
                current_ds = None
                for ds in target_datasets:
                    if ds in rec_name:
                        current_ds = ds
                        break

                def update_tracker(tracker):
                    tracker['sum_abs_err'] += abs_err
                    tracker['sum_err_pct'] += err_pct
                    tracker['count'] += 1

                if current_ds is not None:
                    update_tracker(stats[current_ds])
                
                update_tracker(stats['overall'])

   
    def print_peak_report(title, tracker):
        count = tracker['count']
        if count == 0:
            return
            
        mae_peaks = tracker['sum_abs_err'] / count
        mape_peaks = tracker['sum_err_pct'] / count
        
        print(f"\n{'='*55}")
        print(f"R-PEAK ERROR REPORT: {title.upper()} ({count} segments)")
        print(f"{'='*55}")
        print(f"Mean R-peaks Error (MAE) : {mae_peaks:.4f} peaks/segment")
        print(f"R-peaks Error Percentage : {mape_peaks:.2f} %")
        print(f"{'='*55}")

    for ds in target_datasets:
        print_peak_report(f"DATASET {ds}", stats[ds])
        
    print_peak_report("OVERALL DATASET", stats["overall"])
# def run_peak_count_evaluation():
#     set_seed(SEED)
#     model = load_model()
    
#     try:
#         test_dataset = LoadData(TEST_DATA_PATH)
#         test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) 
#     except Exception as e:
#         print(f"Error loading test dataset: {e}")
#         return

#     total_true_peaks = 0
#     total_pred_peaks = 0
#     total_ratio = 0.0
#     total_samples = 0

#     print("[*] Counting R-peaks using Pan-Tompkins algorithm...")
#     print("[!] USING GROUND TRUTH THRESHOLD FOR PREDICTED ECG")
    
#     with torch.no_grad():
#         for i, batch_data in enumerate(tqdm(test_loader, desc="Counting R-Peaks")):
#             ecg = batch_data[0]
#             ppg = batch_data[1]
            
#             ppg_input = ppg.to(DEVICE).float().unsqueeze(1)
#             ecg_input = ecg.to(DEVICE).float().unsqueeze(1)

#             _, _, predicted_ecg = model(ecg_input, ppg_input)

#             ecg_true_np = np.atleast_2d(ecg.cpu().squeeze().numpy())
#             ecg_pred_np = np.atleast_2d(predicted_ecg.cpu().squeeze().numpy())

#             for b in range(ecg_true_np.shape[0]):
#                 true_s = ecg_true_np[b]
#                 pred_s = ecg_pred_np[b]

#                 num_true, num_pred, ratio = calculate_peak_count_ratio(true_s, pred_s, sampling_rate=125)
                
#                 total_true_peaks += num_true
#                 total_pred_peaks += num_pred
#                 total_ratio += ratio
#                 total_samples += 1

#     avg_ratio = total_ratio / total_samples if total_samples > 0 else 0
#     mae_peaks = abs(total_true_peaks - total_pred_peaks) / total_samples if total_samples > 0 else 0

#     print(f"\n{'='*50}")
#     print(f"R-PEAK COUNT REPORT ({total_samples} samples)")
#     print(f"{'='*50}")
#     print(f"Total Ground Truth R-peaks : {total_true_peaks}")
#     print(f"Total Predicted R-peaks (Pred) : {total_pred_peaks}")
#     print(f"Average Peak Ratio (Pred/True) : {avg_ratio * 100:.2f} %")
#     print(f"Average error per signal : {mae_peaks:.2f} peaks/signal")
#     print(f"{'='*50}")

if __name__ == "__main__":
    # run_visualization()
    # run_loss()
    # run_peak_count_evaluation()
    save_ecg_reconstruction(output_path = '/home/linhhima/PPG_ECG/AF_Detection/reconstructed_ecg/total_mimic_af_ppg2ecg_segment.npz')