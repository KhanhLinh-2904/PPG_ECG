import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import random
import os
from scipy.signal import correlate
from tqdm import tqdm

# Import các module từ dự án của bạn
from CLIP import PPG2ECGModel
from load_data import LoadData
from metric import calculate_cosine_similarity, calculate_dtw_distance, calculate_metrics

# ==========================================
# CONFIGURATION
# ==========================================
SEED = 44
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
OUTPUT_EMBED_DIM = 128
INPUT_LENGTH = 2400

# Đường dẫn dữ liệu và mô hình (Lấy model của Phase 2 vì nó chứa Decoder hoàn chỉnh)
TEST_DATA_PATH = '/home/linhhima/PPG_ECG/datasets/z_score_norm/total_mimic_af.npz'
MODEL_PATH = '/home/linhhima/PPG_ECG/best_multitask_phase2.pth'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_multitask_model():
    """Tải mô hình PPG2ECG và nạp trọng số đã huấn luyện"""
    print(f"Loading PPG2ECG Multi-task Model on {DEVICE}...")
    
    # Khởi tạo mô hình giống với lúc Train
    model = PPG2ECGModel(proj_dim=OUTPUT_EMBED_DIM).to(DEVICE)
    
    # Tải checkpoint
    if torch.cuda.is_available():
        checkpoint = torch.load(MODEL_PATH)
    else:
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        
    # Nạp trọng số
    model.load_state_dict(checkpoint)
    model.eval()
    
    return model

def match_amplitude(pred_s, true_s):
    """
    Kéo dãn biên độ của sóng Generated sao cho 
    độ lệch chuẩn (chênh lệch đỉnh/đáy) bằng với Ground Truth.
    """
    std_true = np.std(true_s)
    std_pred = np.std(pred_s)
    mean_true = np.mean(true_s)
    mean_pred = np.mean(pred_s)
    
    matched_pred = ((pred_s - mean_pred) / (std_pred + 1e-8)) * std_true + mean_true
    return matched_pred

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
def visualize_results(ppg, ecg_true, ecg_pred, sample_idx, record_name):
    print(f"Visualizing Record: {record_name}")
    t_ppg = np.arange(len(ppg))
    t_ecg = np.arange(len(ecg_true))

    plt.figure(figsize=(12, 10))
    plt.suptitle(f"Multi-task Generation | Record: {record_name} - Sample: {sample_idx}", fontsize=14, fontweight='bold')

    plt.subplot(4, 1, 1)
    plt.plot(t_ppg, ppg, color='green', label='Input PPG')
    plt.title("Input PPG Signal")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    plt.subplot(4, 1, 2)
    plt.plot(t_ecg, ecg_true, color='blue', label='Ground Truth ECG')
    plt.title("Ground Truth ECG")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    plt.subplot(4, 1, 3)
    plt.plot(t_ecg, ecg_pred, color='red', label='Generated ECG (From PPG)')
    plt.title("Generated ECG")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    plt.subplot(4, 1, 4)
    plt.plot(t_ecg, ecg_true, color='black', label='Ground Truth', alpha=0.7)
    plt.plot(t_ecg, ecg_pred, color='red', label='Generated', linestyle='--', alpha=0.8)
    plt.title("Comparison: Ground Truth vs Generated ECG")
    plt.xlabel("Time Samples")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    plt.close() 

def run_visualization():
    set_seed(SEED)
    model = load_multitask_model()
    seen_records = set()

    try:
        test_dataset = LoadData(TEST_DATA_PATH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
   
    print("Running Fast Inference for visualization...")
    
    with torch.no_grad():
        for i, (ecg, ppg, record_names, *_) in enumerate(test_loader):
            ppg_input = ppg.to(DEVICE).float().unsqueeze(1)
            
            # ----------------------------------------------------
            # INFERENCE NHANH: Chỉ cần truyền PPG để lấy recon_ecg
            # ----------------------------------------------------
            outputs = model(ppg=ppg_input)
            recon_ecg = outputs["recon_ecg"]

            ppg_np = ppg_input.squeeze(1).cpu().numpy()
            ecg_true_np = ecg.numpy()
            ecg_pred_np = recon_ecg.squeeze(1).cpu().numpy()

            for idx in range(ppg_np.shape[0]):
                current_rec_name = record_names[idx]
                
                if current_rec_name not in seen_records:
                    # Kéo dãn biên độ
                    rescaled_ecg_pred = match_amplitude(ecg_pred_np[idx], ecg_true_np[idx])
                    
                    visualize_results(
                        ppg_np[idx], 
                        ecg_true_np[idx], 
                        rescaled_ecg_pred, 
                        len(seen_records), 
                        current_rec_name
                    )
                    seen_records.add(current_rec_name)
                    
                # Vẽ 5 mẫu biểu đồ rồi dừng
                if len(seen_records) >= 5:
                    return

# ==========================================
# METRICS CALCULATION
# ==========================================
def run_loss():
    model = load_multitask_model()
    
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
            ppg_input = ppg.to(DEVICE).float().unsqueeze(1)

            # Fast Inference
            outputs = model(ppg=ppg_input)
            recon_ecg = outputs["recon_ecg"]

            ecg_true_np = ecg.numpy()
            ecg_pred_np = recon_ecg.squeeze(1).cpu().numpy()

            for b in range(ecg_true_np.shape[0]):
                true_s = ecg_true_np[b]
                
                # Match Amplitude trước khi tính Metric
                pred_s = match_amplitude(ecg_pred_np[b], true_s)

                # Metrics TRƯỚC KHI align
                rmse_b, pearson_b = calculate_metrics(true_s, pred_s)
                dtw_b = calculate_dtw_distance(true_s, pred_s)
                cosine_b = calculate_cosine_similarity(true_s, pred_s)
                
                metrics['before']['rmse'] += rmse_b
                metrics['before']['pearson'] += pearson_b
                metrics['before']['dtw'] += dtw_b
                metrics['before']['cosine'] += cosine_b

                # Metrics SAU KHI align (Loại bỏ sai số trượt pha)
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
    print(f"FINAL MULTI-TASK GENERATION RESULTS ({total_samples} samples)")
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
def save_ecg_reconstruction(output_path="/home/linhhima/PPG_ECG/AF_Detection/total_mimic_af.npz"):
    set_seed(SEED)
    model = load_multitask_model()
    
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
   
    print("Running Inference and saving reconstructions to disk...")
    
    with torch.no_grad():
        for i, (ecg, ppg, record_names, *label) in enumerate(tqdm(test_loader, desc="Saving Batches")):
            ppg_input = ppg.to(DEVICE).float().unsqueeze(1)
            ecg_true_np = ecg.numpy()
            
            outputs = model(ppg=ppg_input)
            recon_ecg = outputs["recon_ecg"]
            pred_ecg_np = recon_ecg.squeeze(1).cpu().numpy()
            
            # Kéo dãn biên độ trước khi lưu
            for b in range(pred_ecg_np.shape[0]):
                pred_ecg_np[b] = match_amplitude(pred_ecg_np[b], ecg_true_np[b])
            
            all_predicted_ecgs.append(pred_ecg_np)
            all_original_ppgs.append(ppg.cpu().numpy())
            if label:
                all_labels.append(label[0].cpu().numpy())
            all_record_names.append(record_names)
            
    save_dict = {
        "ecgs": np.concatenate(all_predicted_ecgs, axis=0),
        "ppgs": np.concatenate(all_original_ppgs, axis=0),
        "records": np.array(all_record_names, dtype=object) 
    }
    if all_labels:
        save_dict["labels"] = np.concatenate(all_labels, axis=0)
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, **save_dict)
    print(f"Successfully saved all reconstructions to {output_path}")

if __name__ == "__main__":
    # run_visualization()
    # run_loss()
    save_ecg_reconstruction()