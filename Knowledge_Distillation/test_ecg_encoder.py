import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import random
import os
from scipy.signal import correlate
from tqdm import tqdm

# Đảm bảo import đúng tên file model và thư viện của bạn
from model import ECG2ECGModel
from load_data import LoadData
from metric import calculate_cosine_similarity, calculate_dtw_distance, calculate_metrics

# ==========================================
# CONFIGURATION
# ==========================================
SEED = 44
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

# Đường dẫn dữ liệu và mô hình
TEST_DATA_PATH = '/home/linhhima/PPG_ECG/datasets/z_score_norm/mimic_III_test.npz'
MODEL_PATH = '/home/linhhima/PPG_ECG/best_ecg2ecg_autoencoder.pth'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_autoencoder():
    """Tải mô hình và nạp trọng số Encoder/Decoder riêng biệt"""
    print(f"Loading ECG2ECG Autoencoder on {DEVICE}...")
    
    # Khởi tạo mô hình
    model = ECG2ECGModel(in_channels=1).to(DEVICE)
    
    # Tải checkpoint
    if torch.cuda.is_available():
        checkpoint = torch.load(MODEL_PATH)
    else:
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        
    # Nạp trọng số từ Dictionary
    model.ecg_enc.load_state_dict(checkpoint['encoder_state_dict'])
    model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    model.eval()
    return model

def align_signals(true_s: np.ndarray, pred_s: np.ndarray) -> np.ndarray:
    """Căn chỉnh pha của 2 tín hiệu bằng Cross-Correlation"""
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

# ==========================================
# VISUALIZATION
# ==========================================
def visualize_results(ecg_true, ecg_pred, sample_idx, record_name):
    """Vẽ đồ thị so sánh tín hiệu gốc và tín hiệu tái tạo"""
    print(f"Visualizing Record: {record_name}")
    t_ecg = np.arange(len(ecg_true))

    plt.figure(figsize=(12, 8))
    plt.suptitle(f"Autoencoder Reconstruction | Record: {record_name} - Sample: {sample_idx}", fontsize=14, fontweight='bold')

    # 1. Ground Truth
    plt.subplot(3, 1, 1)
    plt.plot(t_ecg, ecg_true, color='blue', label='Input ECG (Ground Truth)')
    plt.title("Input ECG Signal")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    # 2. Reconstructed
    plt.subplot(3, 1, 2)
    plt.plot(t_ecg, ecg_pred, color='red', label='Reconstructed ECG')
    plt.title("Reconstructed ECG (Output)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    # 3. Overlay Comparison
    plt.subplot(3, 1, 3)
    plt.plot(t_ecg, ecg_true, color='black', label='Ground Truth', alpha=0.7)
    plt.plot(t_ecg, ecg_pred, color='red', label='Reconstructed', linestyle='--', alpha=0.8)
    plt.title("Comparison Overlay")
    plt.xlabel("Time Samples")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    plt.close() 

def run_visualization():
    set_seed(SEED)
    model = load_autoencoder()
    seen_records = set()

    try:
        test_dataset = LoadData(TEST_DATA_PATH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
   
    print("Running Inference for visualization...")
    
    with torch.no_grad():
        for i, (ecg, ppg, record_names, *_) in enumerate(test_loader):
            ecg_target = ecg.to(DEVICE).float().unsqueeze(1)
            
            # Forward pass (Cực kỳ nhanh)
            outputs = model(ecg=ecg_target)
            recon_ecg = outputs["recon_ecg"]

            ecg_true_np = ecg.numpy()
            ecg_pred_np = recon_ecg.squeeze(1).cpu().numpy()

            for idx in range(ecg_true_np.shape[0]):
                current_rec_name = record_names[idx]
                
                if current_rec_name not in seen_records:
                    visualize_results(
                        ecg_true_np[idx], 
                        ecg_pred_np[idx], 
                        len(seen_records), 
                        current_rec_name
                    )
                    seen_records.add(current_rec_name)
                    
                # Chỉ hiển thị 3-5 hình rồi dừng để tránh spam
                if len(seen_records) >= 3:
                    return

# ==========================================
# METRICS CALCULATION
# ==========================================
def run_loss():
    model = load_autoencoder()
    
    try:
        test_dataset = LoadData(TEST_DATA_PATH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    metrics = {
        'before': {'rmse': 0, 'pearson': 0, 'dtw': 0, 'cosine': 0},
        'after':  {'rmse': 0, 'pearson': 0, 'dtw': 0, 'cosine': 0}
    }
    total_samples = 0

    print("Calculating Metrics over the entire test set...")
    
    with torch.no_grad():
        for i, (ecg, ppg, record_names, *_) in enumerate(tqdm(test_loader, desc="Evaluating")):
            ecg_target = ecg.to(DEVICE).float().unsqueeze(1)

            # Autoencoder Inference
            outputs = model(ecg=ecg_target)
            recon_ecg = outputs["recon_ecg"]

            ecg_true_np = ecg.numpy()
            ecg_pred_np = recon_ecg.squeeze(1).cpu().numpy()

            for b in range(ecg_true_np.shape[0]):
                true_s = ecg_true_np[b]
                pred_s = ecg_pred_np[b]

                # Metrics TRƯỚC KHI align
                rmse_b, pearson_b = calculate_metrics(true_s, pred_s)
                dtw_b = calculate_dtw_distance(true_s, pred_s)
                cosine_b = calculate_cosine_similarity(true_s, pred_s)
                
                metrics['before']['rmse'] += rmse_b
                metrics['before']['pearson'] += pearson_b
                metrics['before']['dtw'] += dtw_b
                metrics['before']['cosine'] += cosine_b

                # Metrics SAU KHI align (Loại bỏ sai số trượt pha nhẹ)
                aligned_pred_s = align_signals(true_s, pred_s)

                rmse_a, pearson_a = calculate_metrics(true_s, aligned_pred_s)
                dtw_a = calculate_dtw_distance(true_s, aligned_pred_s)
                cosine_a = calculate_cosine_similarity(true_s, aligned_pred_s)

                metrics['after']['rmse'] += rmse_a
                metrics['after']['pearson'] += pearson_a
                metrics['after']['dtw'] += dtw_a
                metrics['after']['cosine'] += cosine_a
                
                total_samples += 1

    print(f"\n{'='*55}")
    print(f"FINAL AUTOENCODER RECONSTRUCTION RESULTS ({total_samples} samples)")
    print(f"{'='*55}")
    print(f"{'Metric':<12} | {'Before Align':<16} | {'After Align':<16}")
    print(f"{'-'*55}")
    
    print(f"rRMSE        | {metrics['before']['rmse']/total_samples:<16.4f} | {metrics['after']['rmse']/total_samples:<16.4f}")
    print(f"Pearson      | {metrics['before']['pearson']/total_samples:<16.4f} | {metrics['after']['pearson']/total_samples:<16.4f}")
    print(f"DTW          | {metrics['before']['dtw']/total_samples:<16.4f} | {metrics['after']['dtw']/total_samples:<16.4f}")
    print(f"Cosine       | {metrics['before']['cosine']/total_samples:<16.4f} | {metrics['after']['cosine']/total_samples:<16.4f}")
    print(f"{'='*55}")

# ==========================================
# SAVE TO NUMPY
# ==========================================
def save_ecg_reconstruction(output_path="AF_Detection/autoencoder_ecg_reconstructions.npz"):
    set_seed(SEED)
    model = load_autoencoder()
    
    all_reconstructed_ecgs = []
    all_original_ecgs = []
    all_labels = []
    all_record_names = []

    try:
        test_dataset = LoadData(TEST_DATA_PATH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    except Exception as e:
        print(f"Error: {e}")
        return
   
    print("Running Inference and saving reconstructions to disk...")
    
    with torch.no_grad():
        for i, (ecg, ppg, record_names, *label) in enumerate(tqdm(test_loader, desc="Saving Batches")):
            ecg_target = ecg.to(DEVICE).float().unsqueeze(1)
            
            outputs = model(ecg=ecg_target)
            recon_ecg = outputs["recon_ecg"]
            
            all_reconstructed_ecgs.append(recon_ecg.squeeze(1).cpu().numpy())
            all_original_ecgs.append(ecg.numpy())
            if label:
                all_labels.append(label[0].cpu().numpy())
            all_record_names.append(record_names)
            
    save_dict = {
        "recon_ecgs": np.concatenate(all_reconstructed_ecgs, axis=0),
        "true_ecgs": np.concatenate(all_original_ecgs, axis=0),
        "records": np.array(all_record_names, dtype=object) 
    }
    if all_labels:
        save_dict["labels"] = np.concatenate(all_labels, axis=0)
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, **save_dict)
    print(f"Successfully saved all reconstructions to {output_path}")

if __name__ == "__main__":
    # Bạn có thể bỏ comment hàm nào bạn muốn chạy
    
    # run_visualization()
    run_loss()
    # save_ecg_reconstruction()