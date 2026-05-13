import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import os
import warnings

from scipy.signal import correlate

from load_data import LoadData 
from ecg2ecg import ECGAutoencoder, ECGAEConfig
from ppg2ecg import PPG2ECGModel, PPG2ECGConfig 
from flow_model import LatentRectifiedFlow

# Giả sử bạn có file metric chứa các hàm này
from metric import calculate_cosine_similarity, calculate_dtw_distance, calculate_metrics

# ==========================================
# CẤU HÌNH (CONFIGURATIONS)
# ==========================================
SEED = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
INPUT_LENGTH = 2400
TEST_DATA_PATH = '/home/linhhima/Diffusion datasets/test.npz'

# Đường dẫn trọng số (Cập nhật để bao gồm cả Decoder Finetuned của Phase 3)
PHASE1_ECG_PATH = '/home/linhhima/PPG_ECG/saved_models_ecg_vae/best_ecg_autoencoder.pth'
PHASE1_PPG_PATH = '/home/linhhima/PPG_ECG/saved_models_alignment_batch_32/best_ppg_alignment.pth'
PHASE2_FLOW_PATH = '/home/linhhima/PPG_ECG/saved_models_flow_1024/best_rectified_flow.pth'
PHASE3_DECODER_PATH = '/home/linhhima/PPG_ECG/saved_models_finetune/best_finetuned_decoder.pth' # Đường dẫn mới

ODE_STEPS = 10 # Số bước giải Euler cho Rectified Flow (Giống với lúc train Phase 3)

# ==========================================
# CÁC HÀM TIỆN ÍCH & LOAD MÔ HÌNH
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_models():
    print(f"[*] Loading Models on {DEVICE}...")
    
    # 1. Load ECG Autoencoder (Base Model)
    ecg_cfg = ECGAEConfig(
        input_length=INPUT_LENGTH, in_channels=1, dims=(64, 128, 256, 512),
        depths=(2, 2, 4, 2), latent_channels=16, latent_length=75,
        attn_heads=8, attn_dropout=0.0, global_latent_dim=128, trend_poly=2
    )
    ecg_ae = ECGAutoencoder(ecg_cfg).to(DEVICE)
    ckpt_ecg = torch.load(PHASE1_ECG_PATH, map_location=DEVICE)
    ecg_ae.load_state_dict(ckpt_ecg.get("model_state_dict", ckpt_ecg))
    
    # Ghi đè trọng số của Decoder bằng bản đã Finetune ở Giai đoạn 3
    print(f"[*] Loading Finetuned Decoder from {PHASE3_DECODER_PATH}...")
    ckpt_decoder = torch.load(PHASE3_DECODER_PATH, map_location=DEVICE)
    ecg_ae.decoder.load_state_dict(ckpt_decoder)
    
    ecg_ae.eval()
    for p in ecg_ae.parameters(): p.requires_grad = False

    # 2. Load PPG2ECG Alignment Model
    ppg_cfg = PPG2ECGConfig(
        input_length=INPUT_LENGTH, ppg_in_channels=1, dims=(64, 128, 256, 512),
        depths=(2, 2, 4, 2), latent_channels=16, latent_length=75,
        attn_heads=8, attn_dropout=0.0, use_derivatives=True, proj_dim=128
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
# CÁC CHỨC NĂNG CHÍNH (VISUALIZE, LOSS, SAVE)
# ==========================================

def run_visualization():
    set_seed(SEED)
    ecg_ae, ppg_model, flow_model = load_models()

    try:
        dataset = LoadData(TEST_DATA_PATH)
        # Tắt shuffle để duyệt records theo đúng thứ tự (nếu muốn)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"[*] Visualizing all records in the dataset...")
    
    with torch.no_grad():
        for batch_data in dataloader: # Không cần tqdm vì sẽ bị đứng lại mỗi lần hiển thị ảnh
            ecg = batch_data[0].float().unsqueeze(1).to(DEVICE) if batch_data[0].dim() == 2 else batch_data[0].float().to(DEVICE)
            ppg = batch_data[1].float().unsqueeze(1).to(DEVICE) if batch_data[1].dim() == 2 else batch_data[1].float().to(DEVICE)
            record_names = batch_data[2] if len(batch_data) > 2 else [f"Unknown_Record_{i}" for i in range(ecg.shape[0])]

            # Trích xuất PPG Latent (Condition)
            feat_ppg = ppg_model.ppg_encoder(ppg)
            z_ppg = ppg_model.ppg_latent_head(feat_ppg)
            
            # Sinh Predicted ECG Latent bằng Rectified Flow
            z_ecg_hat = euler_solve(flow_model, z_ppg, num_steps=ODE_STEPS)
            
            # Đưa vào ecg_decoder (Finetuned) để tái tạo sóng ECG
            predicted_ecg = ecg_ae.decoder(z_ecg_hat)

            # --- Hiển thị ngay từng sample trong batch ---
            for idx in range(ppg.shape[0]):
                ppg_plot = ppg[idx].squeeze().cpu().numpy()
                ecg_plot = ecg[idx].squeeze().cpu().numpy()
                pred_plot = predicted_ecg[idx].squeeze().cpu().numpy()
                rec_name = record_names[idx]
                
                t = np.arange(len(ecg_plot))
                
                plt.figure(figsize=(12, 8))
                plt.suptitle(f"Record: {rec_name}", fontsize=14, fontweight='bold')

                plt.subplot(3, 1, 1)
                plt.plot(t, ppg_plot, color='green', label='Input PPG')
                plt.title("Input PPG Signal")
                plt.legend(loc='upper right')

                plt.subplot(3, 1, 2)
                plt.plot(t, ecg_plot, color='blue', label='Ground Truth ECG')
                plt.title("Ground Truth ECG")
                plt.legend(loc='upper right')

                plt.subplot(3, 1, 3)
                plt.plot(t, ecg_plot, color='black', label='Ground Truth', alpha=0.5)
                plt.plot(t, pred_plot, color='red', label='Predicted ECG (Finetuned)', linestyle='--', alpha=0.8)
                plt.title("Comparison: Ground Truth vs Predicted ECG")
                plt.legend(loc='upper right')
                
                plt.tight_layout()
                plt.show() # Code sẽ dừng ở đây chờ bạn tắt cửa sổ ảnh để vẽ tiếp hình sau

def run_loss():
    set_seed(SEED)
    ecg_ae, ppg_model, flow_model = load_models()
    
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

    print("[*] Calculating Metrics...")
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Evaluating Metrics"):
            ecg = batch_data[0].float().unsqueeze(1).to(DEVICE) if batch_data[0].dim() == 2 else batch_data[0].float().to(DEVICE)
            ppg = batch_data[1].float().unsqueeze(1).to(DEVICE) if batch_data[1].dim() == 2 else batch_data[1].float().to(DEVICE)

            feat_ppg = ppg_model.ppg_encoder(ppg)
            z_ppg = ppg_model.ppg_latent_head(feat_ppg)
            z_ecg_hat = euler_solve(flow_model, z_ppg, num_steps=ODE_STEPS)
            predicted_ecg = ecg_ae.decoder(z_ecg_hat)

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
    print(f"FINAL RESULTS ({total_samples} samples)")
    print(f"{'='*40}")
    print(f"{'Metric':<12} | {'Before Align':<12} | {'After Align':<12}")
    print(f"{'-'*40}")
    print(f"RMSE         | {metrics['before']['rmse']/total_samples:<12.4f} | {metrics['after']['rmse']/total_samples:<12.4f}")
    print(f"Pearson      | {metrics['before']['pearson']/total_samples:<12.4f} | {metrics['after']['pearson']/total_samples:<12.4f}")
    print(f"DTW          | {metrics['before']['dtw']/total_samples:<12.4f} | {metrics['after']['dtw']/total_samples:<12.4f}")
    print(f"Cosine       | {metrics['before']['cosine']/total_samples:<12.4f} | {metrics['after']['cosine']/total_samples:<12.4f}")
    print(f"{'='*40}")

def save_ecg_reconstruction(output_path="AF_Detection/ecg_reconstructions_phase3.npz"):
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

if __name__ == "__main__":
    # CHỌN 1 TRONG 3 HÀM ĐỂ CHẠY (Bỏ comment để sử dụng):
    
    run_visualization()
    # run_loss()
    # save_ecg_reconstruction("AF_Detection/total_mimic_af_recon.npz")