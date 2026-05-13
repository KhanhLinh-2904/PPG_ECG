import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from load_data import LoadData
from CLIP_2stage import ECG_PPG_Fusion_Model  
import random
import os
from metric import calculate_cosine_similarity, calculate_dtw_distance, calculate_metrics
from scipy.signal import correlate

# ==========================================
# --- CONFIGURATION ---
# ==========================================
TEST_PHASE = 2  # Đặt = 1 để test Giai đoạn 1 (ECG Autoencoder), Đặt = 2 để test Giai đoạn 2 (PPG -> ECG)

SEED = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
SEQ_LENGTH = 2400
OUTPUT_EMBED_DIM = 128
TEST_DATA_PATH = '/home/linhhima/Diffusion datasets/test.npz'

# Đường dẫn trọng số cho từng giai đoạn
PHASE1_MODEL_PATH = '/home/linhhima/PPG_ECG/best_phase1_reconstruction.pth'
PHASE2_MODEL_PATH = '/home/linhhima/PPG_ECG/best_phase2_fusion.pth'

# ==========================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_model():
    print(f"Loading model for Phase {TEST_PHASE} on {DEVICE}...")
    model = ECG_PPG_Fusion_Model(embed_dim=OUTPUT_EMBED_DIM, freq_dim=256, target_length=SEQ_LENGTH).to(DEVICE)

    model_path = PHASE1_MODEL_PATH if TEST_PHASE == 1 else PHASE2_MODEL_PATH

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy file trọng số: {model_path}")

    if torch.cuda.is_available():
        weights = torch.load(model_path)
    else:
        weights = torch.load(model_path, map_location='cpu')

    model.load_state_dict(weights)
    model.eval()
    return model

def get_prediction(model, ecg_input, ppg_input):
    """ Hàm hỗ trợ inference tùy theo Giai đoạn """
    if TEST_PHASE == 1:
        # Phase 1: ECG -> Encoder ECG -> Decoder -> Reconstructed ECG
        _, z_ecg_3d = model.encode_ecg(ecg_input)
        ecg_pred = model.decoder(z_ecg_3d)
    else:
        # Phase 2: PPG -> Encoder PPG -> Decoder -> Predicted ECG
        _, z_ppg_3d = model.encode_ppg(ppg_input)
        ecg_pred = model.decoder(z_ppg_3d)
    return ecg_pred

def visualize_results(ppg, ecg_true, ecg_pred, record_name):
    print(f"Visualizing Record: {record_name}")
    t_ecg = np.arange(len(ecg_true))

    if TEST_PHASE == 1:
        plt.figure(figsize=(12, 8))
        plt.suptitle(f"[Phase 1: Autoencoder] Record: {record_name}", fontsize=14, fontweight='bold')

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
        plt.plot(t_ecg, ecg_true, color='black', label='Original ECG', alpha=0.7)
        plt.plot(t_ecg, ecg_pred, color='red', label='Reconstructed ECG', linestyle='--', alpha=0.8)
        plt.title("Comparison: Original vs Reconstructed ECG")
        plt.xlabel("Time Samples")
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')

    else:
        t_ppg = np.arange(len(ppg))
        plt.figure(figsize=(12, 10))
        plt.suptitle(f"[Phase 2: Fusion] Record: {record_name}", fontsize=14, fontweight='bold')

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
        plt.plot(t_ecg, ecg_pred, color='red', label='Predicted ECG')
        plt.title("Predicted ECG (Generated from PPG)")
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')

        plt.subplot(4, 1, 4)
        plt.plot(t_ecg, ecg_true, color='black', label='Ground Truth', alpha=0.7)
        plt.plot(t_ecg, ecg_pred, color='red', label='Predicted', linestyle='--', alpha=0.8)
        plt.title("Comparison: Ground Truth vs Predicted ECG")
        plt.xlabel("Time Samples")
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Chỉ hiển thị lên màn hình, KHÔNG LƯU
    plt.show()

def run_visualization():
    set_seed(SEED)
    model = load_model()
    seen_records = set()

    try:
        test_dataset = LoadData(TEST_DATA_PATH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
   
    print(f"Running visualization for Phase {TEST_PHASE}...")
    with torch.no_grad():
        for batch_data in test_loader:
            ecg = batch_data[0]
            ppg = batch_data[1]
            record_names = batch_data[2]

            ppg_input = ppg.to(DEVICE).float().unsqueeze(1)
            ecg_input = ecg.to(DEVICE).float().unsqueeze(1)

            predicted_ecg = get_prediction(model, ecg_input, ppg_input)

            ppg_np = np.atleast_2d(ppg_input.cpu().squeeze().numpy())
            ecg_true_np = np.atleast_2d(ecg.cpu().squeeze().numpy())
            ecg_pred_np = np.atleast_2d(predicted_ecg.cpu().squeeze().numpy())

            for idx in range(ppg_np.shape[0]):
                current_rec_name = record_names[idx]
                
                if current_rec_name not in seen_records:
                    visualize_results(
                        ppg_np[idx], 
                        ecg_true_np[idx], 
                        ecg_pred_np[idx], 
                        current_rec_name
                    )
                    seen_records.add(current_rec_name)
                
                # Chỉ hiển thị 5 ảnh rồi dừng lại
                if len(seen_records) >= 5:
                    print("Đã hiển thị xong 5 samples.")
                    return

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
    set_seed(SEED)
    model = load_model()
    
    try:
        test_dataset = LoadData(TEST_DATA_PATH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    except Exception as e:
        print(f"Error: {e}")
        return

    metrics = {
        'before': {'rmse': 0, 'pearson': 0, 'dtw': 0, 'cosine': 0},
        'after':  {'rmse': 0, 'pearson': 0, 'dtw': 0, 'cosine': 0}
    }
    total_samples = 0

    print(f"Calculating Metrics for Phase {TEST_PHASE}...")
    with torch.no_grad():
        for batch_data in test_loader:
            ecg = batch_data[0]
            ppg = batch_data[1]

            ppg_input = ppg.to(DEVICE).float().unsqueeze(1)
            ecg_input = ecg.to(DEVICE).float().unsqueeze(1)

            predicted_ecg = get_prediction(model, ecg_input, ppg_input)

            ecg_true_np = np.atleast_2d(ecg.cpu().squeeze().numpy())
            ecg_pred_np = np.atleast_2d(predicted_ecg.cpu().squeeze().numpy())

            for b in range(ecg_true_np.shape[0]):
                true_s = ecg_true_np[b]
                pred_s = ecg_pred_np[b]

                rmse_b, pearson_b = calculate_metrics(true_s, pred_s)
                dtw_b = calculate_dtw_distance(true_s, pred_s)
                cosine_b = calculate_cosine_similarity(true_s, pred_s)
                
                metrics['before']['rmse'] += rmse_b
                metrics['before']['pearson'] += pearson_b
                metrics['before']['dtw'] += dtw_b
                metrics['before']['cosine'] += cosine_b

                aligned_pred_s = align_signals(true_s, pred_s)

                rmse_a, pearson_a = calculate_metrics(true_s, aligned_pred_s)
                dtw_a = calculate_dtw_distance(true_s, aligned_pred_s)
                cosine_a = calculate_cosine_similarity(true_s, aligned_pred_s)

                metrics['after']['rmse'] += rmse_a
                metrics['after']['pearson'] += pearson_a
                metrics['after']['dtw'] += dtw_a
                metrics['after']['cosine'] += cosine_a
                
                total_samples += 1

    print(f"\n{'='*40}")
    print(f"FINAL RESULTS - PHASE {TEST_PHASE} ({total_samples} samples)")
    print(f"{'='*40}")
    print(f"{'Metric':<12} | {'Before Align':<12} | {'After Align':<12}")
    print(f"{'-'*40}")
    print(f"RMSE         | {metrics['before']['rmse']/total_samples:<12.4f} | {metrics['after']['rmse']/total_samples:<12.4f}")
    print(f"Pearson      | {metrics['before']['pearson']/total_samples:<12.4f} | {metrics['after']['pearson']/total_samples:<12.4f}")
    print(f"DTW          | {metrics['before']['dtw']/total_samples:<12.4f} | {metrics['after']['dtw']/total_samples:<12.4f}")
    print(f"Cosine       | {metrics['before']['cosine']/total_samples:<12.4f} | {metrics['after']['cosine']/total_samples:<12.4f}")
    print(f"{'='*40}")

def save_ecg_reconstruction(output_path="AF_Detection/ecg_reconstructions.npz"):
    if TEST_PHASE == 1:
        print("CẢNH BÁO: Bạn đang ở Phase 1. Hàm 'save_ecg_reconstruction' chỉ lưu dự đoán PPG -> ECG cho Phase 2.")
        print("Vui lòng thay đổi TEST_PHASE = 2 ở đầu file nếu muốn chạy hàm này.")
        return

    set_seed(SEED)
    model = load_model()
    
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
   
    print("Running inference and saving reconstructions for Phase 2...")
    with torch.no_grad():
        for i, batch_data in enumerate(test_loader):
            ecg = batch_data[0]
            ppg = batch_data[1]
            record_names = batch_data[2]
            label = batch_data[3] if len(batch_data) > 3 else np.zeros(len(ecg))

            ppg_input = ppg.to(DEVICE).float().unsqueeze(1)
            ecg_input = ecg.to(DEVICE).float().unsqueeze(1)

            predicted_ecg = get_prediction(model, ecg_input, ppg_input)
            
            all_predicted_ecgs.append(predicted_ecg.squeeze(1).cpu().numpy())
            all_original_ppgs.append(ppg.cpu().numpy())
            all_labels.append(label.cpu().numpy() if isinstance(label, torch.Tensor) else label)
            all_record_names.append(record_names)
            
    save_dict = {
        "ecgs": np.concatenate(all_predicted_ecgs, axis=0),
        "ppgs": np.concatenate(all_original_ppgs, axis=0),
        "labels": np.concatenate(all_labels, axis=0),
        "records": np.concatenate(all_record_names, axis=0) 
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, **save_dict)
    print(f"Saved reconstructions to {output_path}")

if __name__ == "__main__":
    run_visualization()
    # run_loss()
    # save_ecg_reconstruction("AF_Detection/total_mimic_af.npz")