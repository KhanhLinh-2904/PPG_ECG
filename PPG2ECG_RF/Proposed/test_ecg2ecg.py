import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import random
import os

from load_data import LoadData
from ecg2ecg import ECGAutoencoder, ECGAEConfig 
from metric import calculate_cosine_similarity, calculate_dtw_distance, calculate_metrics
from scipy.signal import correlate

# ==========================================
# --- CONFIGURATION ---
# ==========================================
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
SEQ_LENGTH = 2400

# Test dataset path and model checkpoint weights
TEST_DATA_PATH = '/home/linhhima/Diffusion datasets/combined_segment_split_test.npz'
MODEL_PATH = '/home/linhhima/PPG_ECG/PPG2ECG_RF/Proposed/saved_models_ecg_vae_segment/best_ecg_autoencoder.pth'

# ==========================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_model():
    print(f"[*] Loading ECG Autoencoder model on {DEVICE}...")
    
    cfg = ECGAEConfig(
        input_length=SEQ_LENGTH,
        in_channels=1,
        dims=(64, 128, 256, 512),
        depths=(2, 2, 4, 2),
        latent_channels=16,
        latent_length=75,
        attn_heads=8,
        attn_dropout=0.0,
        global_latent_dim=128,
        trend_poly=2,
    )
    
    model = ECGAutoencoder(cfg).to(DEVICE)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"[!] Model weights file not found at: {MODEL_PATH}")

    # Load checkpoint weights 
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model

def visualize_results(ecg_true, ecg_pred, record_name):
    print(f"Visualizing Record: {record_name}")
    t_ecg = np.arange(len(ecg_true))

    plt.figure(figsize=(12, 8))
    plt.suptitle(f"ECG Autoencoder Reconstruction | Record: {record_name}", fontsize=14, fontweight='bold')

    plt.subplot(3, 1, 1)
    plt.plot(t_ecg, ecg_true, color='blue', label='Input ECG (Ground Truth)')
    plt.title("Input ECG")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    plt.subplot(3, 1, 2)
    plt.plot(t_ecg, ecg_pred, color='red', label='Reconstructed ECG')
    plt.title("Reconstructed ECG")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    plt.subplot(3, 1, 3)
    plt.plot(t_ecg, ecg_true, color='black', label='Ground Truth', alpha=0.7)
    plt.plot(t_ecg, ecg_pred, color='red', label='Reconstructed', linestyle='--', alpha=0.8)
    plt.title("Comparison: Ground Truth vs Reconstructed ECG")
    plt.xlabel("Time Samples")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.show()

def run_visualization():
    set_seed(SEED)
    model = load_model()
    seen_records = set()

    try:
        test_dataset = LoadData(TEST_DATA_PATH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    except Exception as e:
        print(f"[!] Error loading dataset: {e}")
        return
   
    print("\n[*] Running inference for visualization...")
    with torch.no_grad():
        for batch_data in test_loader:
            ecg = batch_data[0]
            record_names = batch_data[2]  # Assuming index 2 contains record names

            # Ensure correct shape [B, 1, 2400]
            ecg_input = ecg.float().unsqueeze(1).to(DEVICE) if ecg.dim() == 2 else ecg.float().to(DEVICE)

            # Autoencoder returns a dictionary
            outputs = model(ecg_input)
            predicted_ecg = outputs["reconstructed_ecg"]

            ecg_true_np = np.atleast_2d(ecg.cpu().squeeze().numpy())
            ecg_pred_np = np.atleast_2d(predicted_ecg.cpu().squeeze().numpy())

            for idx in range(ecg_true_np.shape[0]):
                current_rec_name = record_names[idx]
                
                if current_rec_name not in seen_records:
                    visualize_results(
                        ecg_true_np[idx], 
                        ecg_pred_np[idx], 
                        current_rec_name
                    )
                    seen_records.add(current_rec_name)
                
                # Stop after displaying 5 unique samples
                if len(seen_records) >= 5:
                    print("[*] Displayed 5 demo samples successfully.")
                    return

def align_signals(true_s: np.ndarray, pred_s: np.ndarray) -> np.ndarray:
    """ Uses Cross-Correlation to find the phase lag and shift pred_s to align with true_s. """
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
    set_seed(SEED)
    model = load_model()
    
    try:
        test_dataset = LoadData(TEST_DATA_PATH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    except Exception as e:
        print(f"[!] Error loading dataset: {e}")
        return

    metrics = {
        'before': {'rmse': 0, 'pearson': 0, 'dtw': 0, 'cosine': 0},
        'after':  {'rmse': 0, 'pearson': 0, 'dtw': 0, 'cosine': 0}
    }
    total_samples = 0

    print("\n[*] Calculating performance metrics...")
    with torch.no_grad():
        for batch_data in test_loader:
            ecg = batch_data[0]

            ecg_input = ecg.float().unsqueeze(1).to(DEVICE) if ecg.dim() == 2 else ecg.float().to(DEVICE)

            outputs = model(ecg_input)
            predicted_ecg = outputs["reconstructed_ecg"]

            ecg_true_np = np.atleast_2d(ecg.cpu().squeeze().numpy())
            ecg_pred_np = np.atleast_2d(predicted_ecg.cpu().squeeze().numpy())

            for b in range(ecg_true_np.shape[0]):
                true_s = ecg_true_np[b]
                pred_s = ecg_pred_np[b]

                # 1. BEFORE ALIGNMENT
                rmse_b, pearson_b = calculate_metrics(true_s, pred_s)
                dtw_b = calculate_dtw_distance(true_s, pred_s)
                cosine_b = calculate_cosine_similarity(true_s, pred_s)
                
                metrics['before']['rmse'] += rmse_b
                metrics['before']['pearson'] += pearson_b
                metrics['before']['dtw'] += dtw_b
                metrics['before']['cosine'] += cosine_b

                # 2. SHIFT ALIGNMENT
                aligned_pred_s = align_signals(true_s, pred_s)

                # 3. AFTER ALIGNMENT
                rmse_a, pearson_a = calculate_metrics(true_s, aligned_pred_s)
                dtw_a = calculate_dtw_distance(true_s, aligned_pred_s)
                cosine_a = calculate_cosine_similarity(true_s, aligned_pred_s)

                metrics['after']['rmse'] += rmse_a
                metrics['after']['pearson'] += pearson_a
                metrics['after']['dtw'] += dtw_a
                metrics['after']['cosine'] += cosine_a
                
                total_samples += 1

    print(f"\n{'='*52}")
    print(f"ECG AUTOENCODER EVALUATION REPORT ({total_samples} samples)")
    print(f"{'='*52}")
    print(f"{'Metric':<12} | {'Before Alignment':<17} | {'After Alignment':<17}")
    print(f"{'-'*52}")
    print(f"RMSE         | {metrics['before']['rmse']/total_samples:<17.4f} | {metrics['after']['rmse']/total_samples:<17.4f}")
    print(f"Pearson      | {metrics['before']['pearson']/total_samples:<17.4f} | {metrics['after']['pearson']/total_samples:<17.4f}")
    print(f"DTW          | {metrics['before']['dtw']/total_samples:<17.4f} | {metrics['after']['dtw']/total_samples:<17.4f}")
    print(f"Cosine       | {metrics['before']['cosine']/total_samples:<17.4f} | {metrics['after']['cosine']/total_samples:<17.4f}")
    print(f"{'='*52}")


if __name__ == "__main__":
    
    run_visualization()
    # run_loss()