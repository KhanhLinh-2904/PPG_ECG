import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy.signal import butter, filtfilt, find_peaks
from tqdm import tqdm
import random
import os

from load_data import LoadData
from CLIP import ECG_PPG_Fusion_Model 

# ==========================================
# 1.(CONFIGURATIONS)
# ==========================================
SEED = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
INPUT_LENGTH = 2400
OUTPUT_EMBED_DIM = 128
SEQ_LENGTH = 2400
TEST_DATA_PATH = '/home/linhhima/Diffusion_datasets/combined_segment_split_test.npz'
CLIP_MODEL_PATH = '/home/linhhima/PPG_ECG/best_multitask_model.pth'

SAMPLING_RATE = 125 # Hz
LEFT_WINDOW_MS = 250  # 250ms P wave
RIGHT_WINDOW_MS = 400 # 400ms T wave


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_model():
    print(f"[*] Loading model on {DEVICE}...")
    model = ECG_PPG_Fusion_Model(embed_dim=OUTPUT_EMBED_DIM, freq_dim=256, target_length=SEQ_LENGTH).to(DEVICE)
    weights = torch.load(CLIP_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(weights)
    model.eval()
    return model

def pan_tompkins_qrs(ecg_signal: np.ndarray, fs: int = 125):
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
    # 1. RMSE
    rmse = np.sqrt(np.mean((true_complex - pred_complex) ** 2))
    
    # 2. Pearson Correlation
    std_true = np.std(true_complex)
    std_pred = np.std(pred_complex)
    
    if std_true < 1e-6 or std_pred < 1e-6:
        pearson = 0.0 
    else:
        pearson = np.corrcoef(true_complex, pred_complex)[0, 1]
        
    return rmse, pearson

# ==========================================
# 3. Main for DATASET
# ==========================================
def run_all_complex_evaluation():
    set_seed(SEED)
    model = load_model()
    
    total_rmse = 0.0
    total_pearson = 0.0
    total_valid_beats = 0       
    total_segments_processed = 0 
    
    try:
        test_dataset = LoadData(TEST_DATA_PATH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) 
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print("[*] Starting extraction of ALL heartbeats from ALL Segments...")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Evaluating Beats")):
            ecg = batch_data[0].float().unsqueeze(1).to(DEVICE) if batch_data[0].dim() == 2 else batch_data[0].float().to(DEVICE)
            ppg = batch_data[1].float().unsqueeze(1).to(DEVICE) if batch_data[1].dim() == 2 else batch_data[1].float().to(DEVICE)
            
            _, _, predicted_ecg = model(ecg, ppg)

            ecg_true_np = ecg.cpu().squeeze().numpy()
            ecg_pred_np = predicted_ecg.cpu().squeeze().numpy()

            ecg_true_np = np.atleast_2d(ecg_true_np)
            ecg_pred_np = np.atleast_2d(ecg_pred_np)

            # Scan through each segment in the Batch
            for b in range(ecg_true_np.shape[0]):
                total_segments_processed += 1
                true_s = ecg_true_np[b]
                pred_s = ecg_pred_np[b]

                # STEP 1: Get standard R peaks from ECG Ground Truth
                anchor_r_peaks = pan_tompkins_qrs(true_s, fs=SAMPLING_RATE)

                if len(anchor_r_peaks) == 0:
                    continue # Skip segment if no heartbeat is found

                # STEP 2: Use the same anchor_r_peaks to extract heartbeats for both signals
                # Ensure true_beats and pred_beats are 100% aligned in timeframe
                true_beats = extract_heartbeats(true_s, anchor_r_peaks, SAMPLING_RATE, LEFT_WINDOW_MS, RIGHT_WINDOW_MS)
                pred_beats = extract_heartbeats(pred_s, anchor_r_peaks, SAMPLING_RATE, LEFT_WINDOW_MS, RIGHT_WINDOW_MS)

                # STEP 3: Calculate metrics for EACH HEARTBEAT in the Segment
                for true_complex, pred_complex in zip(true_beats, pred_beats):
                    rmse_val, pearson_val = calculate_complex_metrics(true_complex, pred_complex)
                    
                    # Accumulate
                    total_rmse += rmse_val
                    total_pearson += pearson_val
                    total_valid_beats += 1

    # PRINT FINAL RESULTS REPORT
    if total_valid_beats > 0:
        avg_rmse = total_rmse / total_valid_beats
        avg_pearson = total_pearson / total_valid_beats
        
        print(f"\n{'='*60}")
        print(f"MORPHOLOGY EVALUATION REPORT - ALL BEATS")
        print(f"{'='*60}")
        print(f"Total Segments scanned        : {total_segments_processed:,}")
        print(f"Total valid Beats extracted   : {total_valid_beats:,}")
        print(f"------------------------------------------------------------")
        print(f"Average RMSE (per Beat)       : {avg_rmse:.4f}")
        print(f"Average Pearson (per Beat)    : {avg_pearson:.4f}")
        print(f"{'='*60}")
    else:
        print("\n[!] No valid P-QRS-T complexes found for evaluation.")

if __name__ == "__main__":
    run_all_complex_evaluation()