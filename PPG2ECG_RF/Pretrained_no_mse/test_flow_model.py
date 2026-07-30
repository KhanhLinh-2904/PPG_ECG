import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import os
from scipy.fftpack import fft, fftfreq
from scipy.signal import correlate
from scipy.signal import butter, filtfilt, find_peaks
from load_data import LoadData 
from ecg2ecg import ECGAutoencoder, ECGAEConfig
from ppg2ecg import PPG2ECGModel, PPG2ECGConfig 
from flow_model import LatentRectifiedFlow 

# Metrics utility functions
from metric import calculate_cosine_similarity, calculate_dtw_distance, calculate_metrics, calculate_frechet_distance

# ==========================================
# CONFIGURATIONS
# ==========================================
SEED = 43
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
INPUT_LENGTH = 2400
TEST_DATA_PATH = '/home/linhhima/PPG_ECG/datasets/z_score_norm/total_mimic_af.npz'
# TEST_DATA_PATH = '/home/linhhima/Diffusion_datasets/combined_test.npz'

PHASE1_ECG_PATH = '/home/linhhima/PPG_ECG/PPG2ECG_RF/Proposed/saved_models_ecg_vae_segment/best_ecg_autoencoder.pth'
PHASE1_PPG_PATH = '/home/linhhima/PPG_ECG/PPG2ECG_RF/Pretrained_no_mse/saved_models_alignment_segment/best_ppg_alignment.pth'
PHASE2_FLOW_PATH = '/home/linhhima/PPG_ECG/PPG2ECG_RF/Pretrained_no_mse/saved_models_flow_segment/best_rectified_flow.pth'


ODE_STEPS = 10 # Number of Euler solver steps for Rectified Flow

# ==========================================
# UTILITY FUNCTIONS & MODEL LOADING
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_integrated_energy(signal, fs=125):
    """Auxiliary function to compute integrated energy for threshold plotting."""
    nyq = 0.5 * fs
    b, a = butter(1, [5.0 / nyq, 15.0 / nyq], btype='band')
    filtered = filtfilt(b, a, signal)
    diff = np.diff(filtered)
    diff = np.insert(diff, 0, diff[0])
    squared = diff ** 2
    window = int(0.15 * fs)
    return np.convolve(squared, np.ones(window) / window, mode='same')

def pan_tompkins_qrs(ecg_signal: np.ndarray, fs: int = 125, external_threshold: float = None, return_threshold: bool = False):
    """
    Modified Pan-Tompkins algorithm to accept or return a detection threshold.
    """
    ecg_signal = np.array(ecg_signal).flatten()
    
    # 1. Bandpass Filter (5 - 15 Hz)
    nyq = 0.5 * fs
    low = 5.0 / nyq
    high = 15.0 / nyq
    b, a = butter(1, [low, high], btype='band')
    filtered_ecg = filtfilt(b, a, ecg_signal)
    
    # 2. Derivative
    diff_ecg = np.diff(filtered_ecg)
    diff_ecg = np.insert(diff_ecg, 0, diff_ecg[0])
    
    # 3. Squaring function
    squared_ecg = diff_ecg ** 2
    
    # 4. Moving Window Integration
    window_width = int(0.15 * fs)
    integrated_ecg = np.convolve(squared_ecg, np.ones(window_width) / window_width, mode='same')
    
    # 5. THRESHOLDING
    # Use external threshold if provided; otherwise compute mean
    if external_threshold is not None:
        threshold = external_threshold
    else:
        threshold = np.mean(integrated_ecg)
        
    min_distance = int(0.3 * fs)
    peaks_integrated, _ = find_peaks(integrated_ecg, height=threshold, distance=min_distance)
    
    # 6. Back-search for exact R-peak location on original signal
    r_peaks = []
    search_window = int(0.05 * fs) 
    
    for p in peaks_integrated:
        start = max(0, p - search_window)
        end = min(len(ecg_signal), p + search_window)
        if start < end:
            local_max = np.argmax(ecg_signal[start:end])
            r_peaks.append(start + local_max)
            
    # Return both R-peaks and the applied threshold if requested
    if return_threshold:
        return np.array(r_peaks), threshold
    
    return np.array(r_peaks)

def calculate_peak_count_ratio(true_s: np.ndarray, pred_s: np.ndarray, sampling_rate=125):
    """
    Counts R-peaks using the Pan-Tompkins algorithm and calculates peak ratios.
    NOTE: Enforces Predicted ECG to use the threshold computed from Ground Truth.
    """
    # 1. Detect Ground Truth R-peaks and EXTRACT THRESHOLD (gt_threshold)
    true_peaks, gt_threshold = pan_tompkins_qrs(true_s, fs=sampling_rate, return_threshold=True)
    
    # 2. Pass Ground Truth threshold to Predicted ECG
    pred_peaks = pan_tompkins_qrs(pred_s, fs=sampling_rate, external_threshold=gt_threshold, return_threshold=False)
    
    # 3. Count total peak instances
    num_true = len(true_peaks)
    num_pred = len(pred_peaks)
    
    # 4. Compute prediction ratio relative to Ground Truth
    if num_true > 0:
        ratio = num_pred / num_true
    else:
        ratio = 1.0 if num_pred == 0 else 0.0
        
    return num_true, num_pred, ratio

def load_models():
    print(f"[*] Loading Models on {DEVICE}...")
    
    # 1. Load ECG Autoencoder (Teacher & Decoder)
    ecg_cfg = ECGAEConfig(
        input_length=INPUT_LENGTH, in_channels=1, dims=(64, 128, 256, 512),
        depths=(2, 2, 4, 2), latent_channels=16, latent_length=75,
        attn_heads=8, attn_dropout=0.0, global_latent_dim=128, trend_poly=2
    )
    ecg_ae = ECGAutoencoder(ecg_cfg).to(DEVICE)
    ckpt_ecg = torch.load(PHASE1_ECG_PATH, map_location=DEVICE, weights_only=False)
    ecg_ae.load_state_dict(ckpt_ecg.get("model_state_dict", ckpt_ecg))
    ecg_ae.eval()
    for p in ecg_ae.parameters(): p.requires_grad = False

    # 2. Load PPG2ECG Alignment Model (Extract z_ppg)
    ppg_cfg = PPG2ECGConfig(
        input_length=INPUT_LENGTH, ppg_in_channels=1, dims=(64, 128, 256, 512),
        depths=(2, 2, 4, 2), latent_channels=16, latent_length=75,
        attn_heads=8, attn_dropout=0.0, use_derivatives=True
    )
    ppg_model = PPG2ECGModel(ecg_ae=ecg_ae, cfg=ppg_cfg).to(DEVICE)
    ckpt_ppg = torch.load(PHASE1_PPG_PATH, map_location=DEVICE)
    ppg_model.load_state_dict(ckpt_ppg.get("model_state_dict", ckpt_ppg))
    ppg_model.eval()
    for p in ppg_model.parameters(): p.requires_grad = False

    # 3. Load Stage 2 Latent Rectified Flow
    flow_model = LatentRectifiedFlow(latent_channels=16, cond_channels=16, hidden_dim=128, num_blocks=6).to(DEVICE)
    flow_model.load_state_dict(torch.load(PHASE2_FLOW_PATH, map_location=DEVICE))
    flow_model.eval()
    for p in flow_model.parameters(): p.requires_grad = False
    
    return ecg_ae, ppg_model, flow_model

def euler_solve(flow_model, z_ppg, num_steps=10):
    """ ODE solver function to generate predicted_ecg_latent from z_ppg """
    B, C, L = z_ppg.shape
    xt = torch.randn((B, C, L), device=DEVICE) 
    dt = 1.0 / num_steps
    
    for step in range(num_steps):
        t_val = step * dt
        t_tensor = torch.full((B,), t_val, device=DEVICE)
        v_pred = flow_model(xt, t_tensor, z_ppg)
        xt = xt + v_pred * dt
        
    return xt

def align_signals(true_s: np.ndarray, pred_s: np.ndarray) -> np.ndarray:
    """ 
    Finds phase lag and shifts pred_s to align with true_s using Cross-Correlation.
    """
    correlation = correlate(true_s, pred_s, mode='full')
    lag = np.argmax(correlation) - (len(pred_s) - 1)
    
    aligned_pred = np.empty_like(pred_s) 
    
    if lag > 0:
        aligned_pred[lag:] = pred_s[:-lag]
        edge_value = pred_s[0]
        aligned_pred[:lag] = edge_value
    elif lag < 0:
        lag = abs(lag)
        aligned_pred[:-lag] = pred_s[lag:]
        edge_value = pred_s[-1]
        aligned_pred[-lag:] = edge_value
    else:
        aligned_pred = pred_s.copy()
        
    return aligned_pred

def analyze_p_wave_for_af(signal, fs=125, gt_threshold=None):
    # Pass Ground Truth threshold to peak detection algorithm
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
    
    # Scan from second beat onwards to calculate R-R interval
    for i in range(1, len(r_indices)):
        r = r_indices[i]
        r_prev = r_indices[i-1] 
        
        rr_distance = r - r_prev
        
        # Apply flexible dynamic search window formula
        search_start = int(rr_distance * 0.5)
        search_end = int(0.05 * rr_distance)
        
        start_idx = r - search_start
        end_idx = r - search_end
        
        if start_idx >= r_prev and end_idx < len(signal):
            p_window = clean_signal[start_idx:end_idx]
            p_detrended = p_window - np.mean(p_window)
            
            # --- FIND P-WAVE ---
            peaks, _ = find_peaks(p_detrended, prominence=0.5)
            if len(peaks) > 0:
                num_p_found += 1
                peak_idx_local = peaks[np.argmax(p_detrended[peaks])]
                abs_p_peak = start_idx + peak_idx_local
                p_peaks_indices.append(abs_p_peak)
                
            # --- FFT ENERGY ---
            windowed = p_detrended * np.hamming(len(p_detrended))
            N = len(windowed)
            freqs = fftfreq(N, 1/fs)[:N//2]
            energy = np.abs(fft(windowed))[:N//2] ** 2
            
            power_1_3Hz = np.sum(energy[(freqs >= 1.0) & (freqs < 3.0)])
            power_3_10Hz = np.sum(energy[(freqs >= 3.0)])
            
            total_power_1_3 += power_1_3Hz
            total_power_3_10 += power_3_10Hz

    # Number of analyzed beats (excluding initial beat)
    valid_beats = len(r_indices) - 1
    p_ratio = num_p_found / valid_beats if valid_beats > 0 else 0
    
    if total_power_1_3 == 0: 
        total_power_1_3 = 1e-6 
        
    avg_energy_ratio = (total_power_3_10 / total_power_1_3) if valid_beats > 0 else 0.0
    
    return p_ratio, avg_energy_ratio, p_peaks_indices, r_indices

# ==========================================
# MAIN EXECUTION PIPELINES
# ==========================================
# def run_visualization(max_plots=100):
#     """
#     Plots comparison between reconstructed ECG and Ground Truth ONLY for segments
#     belonging to 'mimic_perform_af_012', while computing and displaying P/R ratio.
#     
#     Args:
#         max_plots (int, optional): Max segments to plot to prevent memory freeze.
#                                    Pass None to plot all without limit.
#     """
#     set_seed(42)
#     ecg_ae, ppg_model, flow_model = load_models()

#     try:
#         dataset = LoadData(TEST_DATA_PATH)
#         # Set shuffle=False to inspect sequentially from start to end of test set
#         dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
#     except Exception as e:
#         print(f"Error loading data: {e}")
#         return

#     # Define target record filter
#     target_record = "mimic_perform_non_af_001"
#     print(f"[*] Visualizing ONLY segments belonging to: {target_record}. Max plots configured: {max_plots}")
#     plot_count = 0
#     
#     with torch.no_grad():
#         for batch_data in dataloader:
#             # Process tensor dimensions and push to device
#             ecg = batch_data[0].float().unsqueeze(1).to(DEVICE) if batch_data[0].dim() == 2 else batch_data[0].float().to(DEVICE)
#             ppg = batch_data[1].float().unsqueeze(1).to(DEVICE) if batch_data[1].dim() == 2 else batch_data[1].float().to(DEVICE)
#             record_names = batch_data[2] if len(batch_data) > 2 else [f"Unknown_{i}" for i in range(ecg.shape[0])]

#             # Feature extraction and flow model prediction
#             feat_ppg = ppg_model.ppg_encoder(ppg)
#             z_ppg = ppg_model.ppg_latent_head(feat_ppg)
#             z_ecg_hat = euler_solve(flow_model, z_ppg, num_steps=ODE_STEPS)
#             predicted_ecg = ecg_ae.decoder(z_ecg_hat)

#             ecg_true_np = np.atleast_2d(ecg.cpu().squeeze().numpy())
#             ecg_pred_np = np.atleast_2d(predicted_ecg.cpu().squeeze().numpy())
#             ppg_np = np.atleast_2d(ppg.cpu().squeeze().numpy())

#             # Process each individual segment in batch
#             for i in range(ecg_true_np.shape[0]):
#                 rec_name = record_names[i]
#                 
#                 # Filter and render only if record name matches target
#                 if target_record not in str(rec_name):
#                     continue  
#                 
#                 ppg_plot = ppg_np[i]
#                 ecg_gt_plot = ecg_true_np[i]
#                 ecg_pred_plot = ecg_pred_np[i]
#                 
#                 # =====================================================================
#                 # COMPUTE P/R RATIO ON RECONSTRUCTED ECG SIGNAL
#                 # =====================================================================
#                 try:
#                     # Step 1: Get standard threshold from Ground Truth ECG (fs=125)
#                     _, gt_threshold = pan_tompkins_qrs(ecg_gt_plot, fs=125, return_threshold=True)
#                     
#                     # Step 2: Analyze P-wave on Reconstructed ECG using gt_threshold
#                     p_ratio, energy_ratio, p_peaks_idx, r_idx = analyze_p_wave_for_af(
#                         ecg_pred_plot, fs=125, gt_threshold=gt_threshold
#                     )
#                     p_ratio_str = f"{p_ratio:.2f}"
#                 except Exception as eval_err:
#                     # Fallback if segment length is too short to complete peak detection
#                     p_ratio_str = "N/A"
#                     p_peaks_idx, r_idx = [], []
#                 
#                 t = np.arange(len(ecg_gt_plot))
#                 time_ax = t / 125  # Convert to time axis in seconds
#                 
#                 # Initialize Figure with 4 subplots
#                 plt.figure(figsize=(12, 11))
#                 plt.suptitle(f"Record: {rec_name} | Reconstructed ECG with P/R Ratio: {p_ratio_str}", fontsize=14, fontweight='bold')

#                 # 1. Input PPG Signal Plot
#                 plt.subplot(4, 1, 1)
#                 plt.plot(t, ppg_plot, color='green', label='Input PPG')
#                 plt.title("Input PPG Signal")
#                 plt.grid(True, alpha=0.3)
#                 plt.legend(loc='upper right')

#                 # 2. Ground Truth ECG Plot
#                 plt.subplot(4, 1, 2)
#                 plt.plot(t, ecg_gt_plot, color='blue', label='Ground Truth ECG')
#                 plt.title("Ground Truth ECG")
#                 plt.grid(True, alpha=0.3)
#                 plt.legend(loc='upper right')

#                 # 3. Predicted ECG (Reconstructed) + Highlighted P & R Peaks
#                 plt.subplot(4, 1, 3)
#                 plt.plot(t, ecg_pred_plot, color='red', label='Predicted ECG')
#                 
#                 # Highlight detected R-peaks and P-peaks on Reconstructed plot
#                 if len(r_idx) > 0:
#                     plt.scatter(r_idx, ecg_pred_plot[r_idx], color='darkred', marker='v', s=60, zorder=3, label='Detected R-Peak')
#                 if len(p_peaks_idx) > 0:
#                     plt.scatter(p_peaks_idx, ecg_pred_plot[p_peaks_idx], color='limegreen', marker='o', s=50, zorder=4, label='Detected P-Peak')
#                 
#                 plt.title("Predicted ECG (Reconstructed)")
#                 plt.grid(True, alpha=0.3)
#                 plt.legend(loc='upper right')

#                 # 4. Comparison Plot (Overlapping signals)
#                 plt.subplot(4, 1, 4)
#                 plt.plot(t, ecg_gt_plot, color='black', label='Ground Truth', alpha=0.5)
#                 plt.plot(t, ecg_pred_plot, color='red', label='Predicted ECG (from Flow)', linestyle='--', alpha=0.8)
#                 plt.title("Comparison: Ground Truth vs Predicted ECG")
#                 plt.grid(True, alpha=0.3)
#                 plt.legend(loc='upper right')
#                 
#                 plt.tight_layout()
#                 plt.subplots_adjust(top=0.92) # Prevent title overlap
#                 plt.show()
#                 
#                 plot_count += 1
#                 
#                 # Check stopping condition
#                 if max_plots is not None and plot_count >= max_plots:
#                     print(f"\n[*] Reached configured limit of {max_plots} segments for {target_record}. Halting!")
#                     return

#     print(f"\n[*] Execution finished. Found and displayed {plot_count} total segments for {target_record}.")

def run_loss():
    set_seed(SEED)
    ecg_ae, ppg_model, flow_model = load_models()
    
    try:
        test_dataset = LoadData(TEST_DATA_PATH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    except Exception as e:
        print(f"Error: {e}")
        return

    # Tracked datasets list
    target_datasets = ["dalia", "wesad", "bidmc", "capno", "mimic"]
    
    # Helper function generating empty metric structure
    def get_empty_metrics():
        return {
            'before': {'rmse': 0.0, 'pearson': 0.0, 'dtw': 0.0, 'cosine': 0.0, 'frechet': 0.0},
            'after':  {'rmse': 0.0, 'pearson': 0.0, 'dtw': 0.0, 'cosine': 0.0, 'frechet': 0.0},
            'count': 0
        }

    # Dictionary storing results per dataset and overall summary
    all_metrics = {ds: get_empty_metrics() for ds in target_datasets}
    all_metrics["overall"] = get_empty_metrics()

    print("[*] Calculating Metrics per Dataset and Overall...")
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Evaluating Metrics"):
            ecg = batch_data[0].float().unsqueeze(1).to(DEVICE) if batch_data[0].dim() == 2 else batch_data[0].float().to(DEVICE)
            ppg = batch_data[1].float().unsqueeze(1).to(DEVICE) if batch_data[1].dim() == 2 else batch_data[1].float().to(DEVICE)
            
            # Retrieve record names to categorize dataset
            record_names = batch_data[2] if len(batch_data) > 2 else [f"unknown_{i}" for i in range(ecg.shape[0])]

            feat_ppg = ppg_model.ppg_encoder(ppg)
            z_ppg = ppg_model.ppg_latent_head(feat_ppg)
            z_ecg_hat = euler_solve(flow_model, z_ppg, num_steps=ODE_STEPS)
            predicted_ecg = ecg_ae.decoder(z_ecg_hat)

            ecg_true_np = np.atleast_2d(ecg.cpu().squeeze().numpy())
            ecg_pred_np = np.atleast_2d(predicted_ecg.cpu().squeeze().numpy())

            for b in range(ecg_true_np.shape[0]):
                true_s = ecg_true_np[b]
                pred_s = ecg_pred_np[b]
                rec_name = str(record_names[b]).lower()

                # 1. Compute metrics before phase alignment
                rmse_b, pearson_b = calculate_metrics(true_s, pred_s)
                dtw_b = calculate_dtw_distance(true_s, pred_s)
                cosine_b = calculate_cosine_similarity(true_s, pred_s)
                frechet_b = calculate_frechet_distance(true_s, pred_s) # Insert Fréchet before align
                
                # Perform phase alignment
                aligned_pred_s = align_signals(true_s, pred_s)
                
                # Compute metrics after phase alignment
                rmse_a, pearson_a = calculate_metrics(true_s, aligned_pred_s)
                dtw_a = calculate_dtw_distance(true_s, aligned_pred_s)
                cosine_a = calculate_cosine_similarity(true_s, aligned_pred_s)
                frechet_a = calculate_frechet_distance(true_s, aligned_pred_s) # Insert Fréchet after align

                # 2. Identify target dataset membership
                current_ds = None
                for ds in target_datasets:
                    if ds in rec_name:
                        current_ds = ds
                        break
                
                # 3. Helper to accumulate metric scores into dictionary
                def add_to_metrics(metric_dict):
                    metric_dict['before']['rmse'] += rmse_b
                    metric_dict['before']['pearson'] += pearson_b
                    metric_dict['before']['dtw'] += dtw_b
                    metric_dict['before']['cosine'] += cosine_b
                    metric_dict['before']['frechet'] += frechet_b
                    
                    metric_dict['after']['rmse'] += rmse_a
                    metric_dict['after']['pearson'] += pearson_a
                    metric_dict['after']['dtw'] += dtw_a
                    metric_dict['after']['cosine'] += cosine_a
                    metric_dict['after']['frechet'] += frechet_a
                    
                    metric_dict['count'] += 1

                # Accumulate into matching dataset dictionary
                if current_ds is not None:
                    add_to_metrics(all_metrics[current_ds])
                
                # Always accumulate into Overall dictionary
                add_to_metrics(all_metrics["overall"])

    # ==========================================
    # PRINT PERFORMANCE REPORT
    # ==========================================
    def print_report(title, m_dict):
        count = m_dict['count']
        if count == 0:
            return # Skip if dataset has no samples in test set
            
        print(f"\n{'='*55}")
        print(f"RESULTS: {title.upper()} ({count} samples)")
        print(f"{'='*55}")
        print(f"{'Metric':<12} | {'Before Align':<15} | {'After Align':<15}")
        print(f"{'-'*55}")
        print(f"RMSE         | {m_dict['before']['rmse']/count:<15.4f} | {m_dict['after']['rmse']/count:<15.4f}")
        print(f"Pearson      | {m_dict['before']['pearson']/count:<15.4f} | {m_dict['after']['pearson']/count:<15.4f}")
        print(f"DTW          | {m_dict['before']['dtw']/count:<15.4f} | {m_dict['after']['dtw']/count:<15.4f}")
        print(f"Cosine       | {m_dict['before']['cosine']/count:<15.4f} | {m_dict['after']['cosine']/count:<15.4f}")
        print(f"Fréchet      | {m_dict['before']['frechet']/count:<15.4f} | {m_dict['after']['frechet']/count:<15.4f}")
        print(f"{'='*55}")

    # Print report per dataset
    for ds in target_datasets:
        print_report(f"DATASET: {ds}", all_metrics[ds])
        
    # Print overall dataset report
    print_report("OVERALL DATASET", all_metrics["overall"])

def save_ecg_reconstruction(output_path="AF_Detection/ecg_reconstructions.npz"):
    set_seed(SEED)
    ecg_ae, ppg_model, flow_model = load_models()
    
    all_predicted_ecgs = []
    all_original_ppgs = []
    all_labels = []
    all_record_names = []

    try:
        test_dataset = LoadData(TEST_DATA_PATH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    except Exception as e:
        print(f"Error: {e}")
        return
   
    print(f"[*] Running inference and saving reconstructions to {output_path}...")
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Saving Data"):
            ecg = batch_data[0]
            ppg = batch_data[1]
            record_names = batch_data[2] if len(batch_data) > 2 else []
            label = batch_data[3] if len(batch_data) > 3 else np.zeros(len(ecg))

            ppg_input = ppg.to(DEVICE).float().unsqueeze(1) if ppg.dim() == 2 else ppg.float().to(DEVICE)
            
            feat_ppg = ppg_model.ppg_encoder(ppg_input)
            z_ppg = ppg_model.ppg_latent_head(feat_ppg)
            z_ecg_hat = euler_solve(flow_model, z_ppg, num_steps=ODE_STEPS)
            
            predicted_ecg = ecg_ae.decoder(z_ecg_hat)
            
            all_predicted_ecgs.append(predicted_ecg.squeeze(1).cpu().numpy())
            all_original_ppgs.append(ppg.cpu().numpy())
            all_labels.append(label.cpu().numpy() if isinstance(label, torch.Tensor) else label)
            all_record_names.append(record_names)
            
    save_dict = {
        "ecgs": np.concatenate(all_predicted_ecgs, axis=0),
        "ppgs": np.concatenate(all_original_ppgs, axis=0),
        "labels": np.concatenate(all_labels, axis=0),
        "records": np.concatenate(all_record_names, axis=0) if all_record_names else np.array([])
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, **save_dict)
    print(f"[+] Saved reconstructions successfully to {output_path}")

def run_peak_count_evaluation():
    set_seed(SEED)
    # Load Stage 1 & Stage 2 models for Flow inference
    ecg_ae, ppg_model, flow_model = load_models()
    
    try:
        test_dataset = LoadData(TEST_DATA_PATH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) 
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        return

    # 1. Initialize tracker dictionaries per dataset
    target_datasets = ["dalia", "wesad", "bidmc", "capno", "mimic"]
    
    def get_empty_tracker():
        return {
            'sum_abs_err': 0.0, 
            'sum_err_pct': 0.0, 
            'count': 0
        }
        
    stats = {ds: get_empty_tracker() for ds in target_datasets}
    stats["overall"] = get_empty_tracker()

    print("[*] Counting R-peaks and computing error metrics (MAE & Error Percentage)...")
    print("[!] WARNING: Enforcing Predicted ECG to share detection threshold with Ground Truth!")
    
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Evaluating R-Peaks Errors"):
            ecg = batch_data[0].float().unsqueeze(1).to(DEVICE) if batch_data[0].dim() == 2 else batch_data[0].float().to(DEVICE)
            ppg = batch_data[1].float().unsqueeze(1).to(DEVICE) if batch_data[1].dim() == 2 else batch_data[1].float().to(DEVICE)
            
            # Retrieve record_names for dataset classification
            record_names = batch_data[2] if len(batch_data) > 2 else [f"unknown_{j}" for j in range(ecg.shape[0])]

            # Feature extraction and prediction via flow model
            feat_ppg = ppg_model.ppg_encoder(ppg)
            z_ppg = ppg_model.ppg_latent_head(feat_ppg)
            z_ecg_hat = euler_solve(flow_model, z_ppg, num_steps=ODE_STEPS)
            predicted_ecg = ecg_ae.decoder(z_ecg_hat)

            ecg_true_np = np.atleast_2d(ecg.cpu().squeeze().numpy())
            ecg_pred_np = np.atleast_2d(predicted_ecg.cpu().squeeze().numpy())

            for b in range(ecg_true_np.shape[0]):
                true_s = ecg_true_np[b]
                pred_s = ecg_pred_np[b]
                rec_name = str(record_names[b]).lower()

                # Get peak counts via algorithm
                num_true, num_pred, _ = calculate_peak_count_ratio(true_s, pred_s, sampling_rate=125)
                
                # =========================================================
                # COMPUTE SEGMENT ERROR METRICS
                # =========================================================
                abs_err = abs(num_true - num_pred)
                err_pct = (abs_err / num_true * 100) if num_true > 0 else 0.0
                
                # Identify matching dataset
                current_ds = None
                for ds in target_datasets:
                    if ds in rec_name:
                        current_ds = ds
                        break

                # Helper to update tracker statistics
                def update_tracker(tracker):
                    tracker['sum_abs_err'] += abs_err
                    tracker['sum_err_pct'] += err_pct
                    tracker['count'] += 1

                # Update matching dataset statistics
                if current_ds is not None:
                    update_tracker(stats[current_ds])
                
                # Always update overall statistics
                update_tracker(stats['overall'])

    # ==========================================
    # PRINT PERFORMANCE REPORT
    # ==========================================
    def print_peak_report(title, tracker):
        count = tracker['count']
        if count == 0:
            return
            
        # Compute mean across total segments
        mae_peaks = tracker['sum_abs_err'] / count
        mape_peaks = tracker['sum_err_pct'] / count
        
        print(f"\n{'='*55}")
        print(f"R-PEAK ERROR REPORT: {title.upper()} ({count} segments)")
        print(f"{'='*55}")
        print(f"Mean R-peaks Error (MAE) : {mae_peaks:.4f} peaks/segment")
        print(f"R-peaks Error Percentage : {mape_peaks:.2f} %")
        print(f"{'='*55}")

    # Print report per dataset
    for ds in target_datasets:
        print_peak_report(f"DATASET {ds}", stats[ds])
        
    # Print overall dataset report
    print_peak_report("OVERALL DATASET", stats["overall"])

if __name__ == "__main__":
    # CHOOSE 1 OF THE FUNCTIONS TO RUN (Uncomment to execute):
    
    # run_visualization()
    run_loss()
    # save_ecg_reconstruction("/home/linhhima/PPG_ECG/AF_Detection/reconstructed_ecg/total_mimic_af_flow_segment.npz")
    # run_peak_count_evaluation()