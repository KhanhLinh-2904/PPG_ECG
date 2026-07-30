import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import os

from scipy.signal import butter, filtfilt, find_peaks

# Import internal modules
from load_data import LoadData 
from ecg2ecg import ECGAutoencoder, ECGAEConfig
from ppg2ecg import PPG2ECGModel, PPG2ECGConfig 
from flow_model import LatentRectifiedFlow 

# ==========================================
# 1. CONFIGURATIONS
# ==========================================
SEED = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
INPUT_LENGTH = 2400
TEST_DATA_PATH = '/home/linhhima/Diffusion_datasets/combined_segment_split_test.npz'

# Checkpoint paths
PHASE1_ECG_PATH = '/home/linhhima/PPG_ECG/PPG2ECG_RF/Proposed/saved_models_ecg_vae_segment/best_ecg_autoencoder.pth'
PHASE1_PPG_PATH = '/home/linhhima/PPG_ECG/PPG2ECG_RF/Proposed/saved_models_alignment_segment/best_ppg_alignment.pth'
PHASE2_FLOW_PATH = '/home/linhhima/PPG_ECG/PPG2ECG_RF/Proposed/saved_models_flow_segment/best_rectified_flow.pth'

ODE_STEPS = 10 # Number of Euler steps for Rectified Flow

# P-QRS-T complex extraction parameters
SAMPLING_RATE = 125 # Hz
LEFT_WINDOW_MS = 250  # Look back 250 ms to capture the P-wave
RIGHT_WINDOW_MS = 400 # Look forward 400 ms to capture the T-wave

# ==========================================
# 2. UTILITY FUNCTIONS & MODEL LOADING
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_models():
    print(f"[*] Loading Diffusion Models on {DEVICE}...")
    
    # 1. Load ECG Autoencoder
    ecg_cfg = ECGAEConfig(
        input_length=INPUT_LENGTH, in_channels=1, dims=(64, 128, 256, 512),
        depths=(2, 2, 4, 2), latent_channels=16, latent_length=75,
        attn_heads=8, attn_dropout=0.0, global_latent_dim=128, trend_poly=2
    )
    ecg_ae = ECGAutoencoder(ecg_cfg).to(DEVICE)
    ckpt_ecg = torch.load(PHASE1_ECG_PATH, map_location=DEVICE)
    ecg_ae.load_state_dict(ckpt_ecg.get("model_state_dict", ckpt_ecg))
    ecg_ae.eval()
    for p in ecg_ae.parameters(): p.requires_grad = False

    # 2. Load PPG2ECG Alignment Model
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

    # 3. Load Latent Rectified Flow
    flow_model = LatentRectifiedFlow(latent_channels=16, cond_channels=16, hidden_dim=128, num_blocks=6).to(DEVICE)
    flow_model.load_state_dict(torch.load(PHASE2_FLOW_PATH, map_location=DEVICE))
    flow_model.eval()
    for p in flow_model.parameters(): p.requires_grad = False
    
    return ecg_ae, ppg_model, flow_model

def euler_solve(flow_model, z_ppg, num_steps=10):
    """ODE solver function to generate predicted_ecg_latent."""
    B, C, L = z_ppg.shape
    xt = torch.randn((B, C, L), device=DEVICE) 
    dt = 1.0 / num_steps
    for step in range(num_steps):
        t_val = step * dt
        t_tensor = torch.full((B,), t_val, device=DEVICE)
        v_pred = flow_model(xt, t_tensor, z_ppg)
        xt = xt + v_pred * dt
    return xt

def pan_tompkins_qrs(ecg_signal: np.ndarray, fs: int = 125):
    """Detect reference R-peaks from the Ground Truth signal."""
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

def extract_heartbeats(ecg_signal: np.ndarray, r_peaks: np.ndarray, fs: int, left_ms: int, right_ms: int):
    """Extract P-QRS-T segments around the reference R-peaks."""
    ecg_signal = np.array(ecg_signal).flatten()
    left_samples = int((left_ms / 1000.0) * fs)
    right_samples = int((right_ms / 1000.0) * fs)
    
    heartbeats = []
    for r in r_peaks:
        if r - left_samples >= 0 and r + right_samples < len(ecg_signal):
            beat = ecg_signal[r - left_samples : r + right_samples]
            heartbeats.append(beat)
            
    return np.array(heartbeats)

def calculate_complex_metrics(true_complex, pred_complex):
    """Calculate RMSE and Pearson correlation coefficient for a single heartbeat pair."""
    rmse = np.sqrt(np.mean((true_complex - pred_complex) ** 2))
    
    std_true = np.std(true_complex)
    std_pred = np.std(pred_complex)
    
    if std_true < 1e-6 or std_pred < 1e-6:
        pearson = 0.0 
    else:
        pearson = np.corrcoef(true_complex, pred_complex)[0, 1]
        
    return rmse, pearson

# ==========================================
# 3. MAIN FUNCTION: EXTRACTION & AVERAGE METRICS
# ==========================================
def run_diffusion_all_complex_evaluation():
    set_seed(SEED)
    ecg_ae, ppg_model, flow_model = load_models()
    
    # PLOT CONTROL FLAG (Keep False for fast evaluation over the entire dataset)
    SHOW_PLOTS = False 
    
    # Cumulative metrics tracking
    total_rmse = 0.0
    total_pearson = 0.0
    total_valid_beats = 0        # Count total processed heartbeats
    total_segments_processed = 0 # Count total evaluated segments

    try:
        test_dataset = LoadData(TEST_DATA_PATH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) 
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print("[*] Starting Diffusion inference and extracting ALL heartbeats from ALL Segments...")
    
    total_samples_time = int((LEFT_WINDOW_MS + RIGHT_WINDOW_MS) / 1000.0 * SAMPLING_RATE)
    time_axis_ms = np.linspace(-LEFT_WINDOW_MS, RIGHT_WINDOW_MS, total_samples_time)

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Evaluating Beats")):
            ecg = batch_data[0].float().unsqueeze(1).to(DEVICE) if batch_data[0].dim() == 2 else batch_data[0].float().to(DEVICE)
            ppg = batch_data[1].float().unsqueeze(1).to(DEVICE) if batch_data[1].dim() == 2 else batch_data[1].float().to(DEVICE)
            
            record_names = batch_data[2] if len(batch_data) > 2 else [f"Batch_{batch_idx}_Idx_{i}" for i in range(ecg.shape[0])]

            # --- DIFFUSION MODEL INFERENCE PIPELINE ---
            feat_ppg = ppg_model.ppg_encoder(ppg)
            z_ppg = ppg_model.ppg_latent_head(feat_ppg)
            z_ecg_hat = euler_solve(flow_model, z_ppg, num_steps=ODE_STEPS)
            predicted_ecg = ecg_ae.decoder(z_ecg_hat)
            # ------------------------------------------

            ecg_true_np = np.atleast_2d(ecg.cpu().squeeze().numpy())
            ecg_pred_np = np.atleast_2d(predicted_ecg.cpu().squeeze().numpy())

            # Iterate over each segment in the current Batch
            for b in range(ecg_true_np.shape[0]):
                total_segments_processed += 1
                rec_name = record_names[b]

                true_s = ecg_true_np[b]
                pred_s = ecg_pred_np[b]

                # STEP 1: Detect reference R-peaks from Ground Truth ECG
                anchor_r_peaks = pan_tompkins_qrs(true_s, fs=SAMPLING_RATE)

                if len(anchor_r_peaks) == 0:
                    continue # Skip segment if no heartbeats are detected

                # STEP 2: Use anchor_r_peaks to extract matching heartbeats (Ensures time-alignment)
                true_beats = extract_heartbeats(true_s, anchor_r_peaks, SAMPLING_RATE, LEFT_WINDOW_MS, RIGHT_WINDOW_MS)
                pred_beats = extract_heartbeats(pred_s, anchor_r_peaks, SAMPLING_RATE, LEFT_WINDOW_MS, RIGHT_WINDOW_MS)

                # STEP 3: Iterate through all extracted beats
                for beat_idx, (true_complex, pred_complex) in enumerate(zip(true_beats, pred_beats)):
                    rmse_val, pearson_val = calculate_complex_metrics(true_complex, pred_complex)
                    
                    # Accumulate totals
                    total_rmse += rmse_val
                    total_pearson += pearson_val
                    total_valid_beats += 1

                    # Plotting (Only display the first 5 sample plots if SHOW_PLOTS is True)
                    if SHOW_PLOTS and total_valid_beats <= 5:
                        plt.figure(figsize=(10, 6))
                        plt.plot(time_axis_ms, true_complex, color='black', linewidth=2.5, label='Ground Truth Complex')
                        plt.plot(time_axis_ms, pred_complex, color='red', linestyle='--', linewidth=2, label='Predicted Complex')
                        
                        plt.axvline(x=0, color='gray', linestyle=':', alpha=0.7, label='R-peak Center (0 ms)')

                        plt.title(f"Record: {rec_name} | Beat: {beat_idx + 1}\nRectified Flow Model: Single P-QRS-T Complex", fontsize=14, fontweight='bold')
                        plt.suptitle(f"RMSE: {rmse_val:.4f}  |  Pearson: {pearson_val:.4f}", color='blue', fontsize=12)
                        plt.xlabel("Time relative to R-peak (ms)", fontsize=11)
                        plt.ylabel("Amplitude", fontsize=11)
                        plt.legend(loc="upper right")
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.show()

    # PRINT FINAL OVERALL EVALUATION REPORT
    if total_valid_beats > 0:
        avg_rmse = total_rmse / total_valid_beats
        avg_pearson = total_pearson / total_valid_beats
        
        print(f"\n{'='*60}")
        print(f"MORPHOLOGICAL EVALUATION REPORT - RECTIFIED FLOW MODEL")
        print(f"{'='*60}")
        print(f"Total Processed Segments          : {total_segments_processed:,}")
        print(f"Total Valid Beats Extracted       : {total_valid_beats:,}")
        print(f"------------------------------------------------------------")
        print(f"Average RMSE (per Beat)           : {avg_rmse:.4f}")
        print(f"Average Pearson Correlation       : {avg_pearson:.4f}")
        print(f"{'='*60}")
    else:
        print("\n[!] No valid P-QRS-T complexes were found in the dataset.")

if __name__ == "__main__":
    run_diffusion_all_complex_evaluation()