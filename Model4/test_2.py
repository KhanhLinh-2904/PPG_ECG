import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import random
import os
from scipy.signal import correlate

from load_data import LoadData
from metric import calculate_cosine_similarity, calculate_dtw_distance, calculate_metrics

# Đảm bảo bạn import đúng tên Class model mới từ file chứa model (ví dụ CLIP.py)
from CLIP import PPGtoECGDualBranchReconstructionNet

# --- CONFIGURATION ---
SEED = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
INPUT_LENGTH = 2400

# Các config cấu trúc mô hình phải khớp 100% với lúc training
MODEL_DIMS = (48, 96, 128, 256)
NUM_BLOCKS_PER_STAGE = 2
NUM_HEADS = 8
ATTN_DEPTH_QRS = 2
ATTN_DEPTH_NON_QRS = 2

TEST_DATA_PATH = '/home/linhhima/Diffusion datasets/test.npz'
# Đã đổi sang file weights lưu từ mô hình Dual-Branch mới nhất
CLIP_MODEL_PATH = '/home/linhhima/PPG_ECG/best_dual_branch_model.pth' 


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_model():
    print(f"Loading model on {DEVICE}...")
    
    # Khởi tạo mô hình kiến trúc nhánh kép (Dual Branch)
    model = PPGtoECGDualBranchReconstructionNet(
        input_len=INPUT_LENGTH,
        dims=MODEL_DIMS,
        num_blocks_per_stage=NUM_BLOCKS_PER_STAGE,
        num_heads=NUM_HEADS,
        attn_depth_qrs=ATTN_DEPTH_QRS,
        attn_depth_non_qrs=ATTN_DEPTH_NON_QRS,
        drop_path=0.0  # Lúc test (inference) thì luôn để drop_path = 0.0
    ).to(DEVICE)

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
            
            # Thay đổi hàm gọi Model và key lấy tín hiệu
            outputs = model(ppg_raw=ppg_input)
            predicted_ecg = outputs["ecg_final_pred"]
            
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

def visualize_results(ppg, ecg_true, ecg_pred, sample_idx, record_name):
    print(f"Visualizing Record: {record_name}")
    t_ppg = np.arange(len(ppg))
    t_ecg = np.arange(len(ecg_true))

    plt.figure(figsize=(12, 10))
    plt.suptitle(f"Record: {record_name} - Sample ID: {sample_idx}", fontsize=14, fontweight='bold')

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
    plt.title("Predicted ECG (Reconstructed)")
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
    plt.show()

def run_visualization():
    set_seed(SEED)
    model = load_model()
    seen_records = set()

    try:
        test_dataset = LoadData(TEST_DATA_PATH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    except Exception as e:
        print(f"Error: {e}")
        return
   
    print("Running inference for visualization...")
    with torch.no_grad():
        for i, (ecg, ppg, record_names) in enumerate(test_loader):
            ppg_input = ppg.to(DEVICE).float().unsqueeze(1)
            
            # Cập nhật lời gọi Model
            outputs = model(ppg_raw=ppg_input)
            predicted_ecg = outputs["ecg_final_pred"]

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
                        len(seen_records), 
                        current_rec_name
                    )
                    seen_records.add(current_rec_name)

def align_signals(true_s: np.ndarray, pred_s: np.ndarray) -> np.ndarray:
    """
    Dùng Cross-Correlation để tìm độ lệch pha và dịch chuyển pred_s cho khớp với true_s.
    """
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

    metrics = {
        'before': {'rmse': 0, 'pearson': 0, 'dtw': 0, 'cosine': 0},
        'after':  {'rmse': 0, 'pearson': 0, 'dtw': 0, 'cosine': 0}
    }
    total_samples = 0

    print("Calculating Metrics...")
    with torch.no_grad():
        for i, (ecg, ppg, record_names) in enumerate(test_loader):
            ppg_input = ppg.to(DEVICE).float().unsqueeze(1)

            # Cập nhật lời gọi Model
            outputs = model(ppg_raw=ppg_input)
            predicted_ecg = outputs["ecg_final_pred"]

            ecg_true_np = np.atleast_2d(ecg.cpu().squeeze().numpy())
            ecg_pred_np = np.atleast_2d(predicted_ecg.cpu().squeeze().numpy())

            for b in range(ecg_true_np.shape[0]):
                true_s = ecg_true_np[b]
                pred_s = ecg_pred_np[b]

                # ---------------------------------------------------------
                # 1. TÍNH METRICS TRƯỚC KHI SHIFT ALIGNMENT
                # ---------------------------------------------------------
                rmse_b, pearson_b = calculate_metrics(true_s, pred_s)
                dtw_b = calculate_dtw_distance(true_s, pred_s)
                cosine_b = calculate_cosine_similarity(true_s, pred_s)
                
                metrics['before']['rmse'] += rmse_b
                metrics['before']['pearson'] += pearson_b
                metrics['before']['dtw'] += dtw_b
                metrics['before']['cosine'] += cosine_b

                # ---------------------------------------------------------
                # 2. DỊCH CHUYỂN PHA (SHIFT ALIGNMENT)
                # ---------------------------------------------------------
                aligned_pred_s = align_signals(true_s, pred_s)

                # ---------------------------------------------------------
                # 3. TÍNH METRICS SAU KHI SHIFT ALIGNMENT
                # ---------------------------------------------------------
                rmse_a, pearson_a = calculate_metrics(true_s, aligned_pred_s)
                dtw_a = calculate_dtw_distance(true_s, aligned_pred_s)
                cosine_a = calculate_cosine_similarity(true_s, aligned_pred_s)

                metrics['after']['rmse'] += rmse_a
                metrics['after']['pearson'] += pearson_a
                metrics['after']['dtw'] += dtw_a
                metrics['after']['cosine'] += cosine_a
                
                total_samples += 1

    print(f"\n{'='*40}")
    print(f"FINAL RESULTS ({total_samples} samples)")
    print(f"{'='*40}")
    print(f"{'Metric':<12} | {'Before Align':<12} | {'After Align':<12}")
    print(f"{'-'*40}")
    
    print(f"rRMSE        | {metrics['before']['rmse']/total_samples:<12.4f} | {metrics['after']['rmse']/total_samples:<12.4f}")
    print(f"Pearson      | {metrics['before']['pearson']/total_samples:<12.4f} | {metrics['after']['pearson']/total_samples:<12.4f}")
    print(f"DTW          | {metrics['before']['dtw']/total_samples:<12.4f} | {metrics['after']['dtw']/total_samples:<12.4f}")
    print(f"Cosine       | {metrics['before']['cosine']/total_samples:<12.4f} | {metrics['after']['cosine']/total_samples:<12.4f}")
    print(f"{'='*40}")

def save_ecg_reconstruction_deepbeat(output_path="AF_Detection/ecg_deepbeat_reconstructions.npz"):
    set_seed(SEED)
    model = load_model()
    
    all_predicted_ecgs = []
    all_original_ppgs = []
    all_labels = []

    try:
        test_dataset = LoadData(TEST_DATA_PATH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    except Exception as e:
        print(f"Error: {e}")
        return
   
    print("Running inference and saving DeepBeat reconstructions...")
    with torch.no_grad():
        for i, (ppg, label) in enumerate(test_loader):
            ppg_input = ppg.to(DEVICE).float().unsqueeze(1)
            
            # Cập nhật lời gọi Model
            outputs = model(ppg_raw=ppg_input)
            predicted_ecg = outputs["ecg_final_pred"]
            
            all_predicted_ecgs.append(predicted_ecg.squeeze(1).cpu().numpy())
            all_original_ppgs.append(ppg.cpu().numpy())
            all_labels.append(label.cpu().numpy())
            
    save_dict = {
        "ecgs": np.concatenate(all_predicted_ecgs, axis=0),
        "ppgs": np.concatenate(all_original_ppgs, axis=0),
        "labels": np.concatenate(all_labels, axis=0),
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, **save_dict)
    print(f"Saved DeepBeat reconstructions to {output_path}")

if __name__ == "__main__":
    # run_visualization()
    run_loss()
    # save_ecg_reconstruction("AF_Detection/total_mimic_af.npz")
    # save_ecg_reconstruction_deepbeat("AF_Detection/deepbeat_ecg_reconstructions.npz")