import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import os
import warnings

# Thêm thư viện hỗ trợ vẽ 3D và tính toán phân phối Gaussian KDE/Lý thuyết
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats
from matplotlib.patches import Patch

from load_data import LoadData 
from ecg2ecg import ECGAutoencoder, ECGAEConfig
from ppg2ecg import PPG2ECGModel, PPG2ECGConfig 
from flow_model import LatentRectifiedFlow # Đảm bảo bạn có file chứa class này

# ==========================================
# CẤU HÌNH (CONFIGURATIONS)
# ==========================================
SEED = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
INPUT_LENGTH = 2400
TEST_DATA_PATH = '/home/linhhima/PPG_ECG/datasets/z_score_norm/total_mimic_af.npz'

# Đường dẫn trọng số
PHASE1_ECG_PATH = '/home/linhhima/PPG_ECG/Result_model2/saved_models_ecg_vae/best_ecg_autoencoder.pth'
PHASE1_PPG_PATH = '/home/linhhima/PPG_ECG/Result_model2/saved_models_alignment_batch_32/best_ppg_alignment.pth'
PHASE2_FLOW_PATH = '/home/linhhima/PPG_ECG/saved_models_flow_cfg/best_rectified_flow_cfg.pth' # Đường dẫn mô hình Flow

NUM_SAMPLES_TO_VISUALIZE = 2400
ODE_STEPS = 10 # Số bước giải Euler cho Rectified Flow

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_models():
    print(f"[*] Loading Models on {DEVICE}...")
    
    # 1. Load ECG Autoencoder (Teacher & Decoder)
    ecg_cfg = ECGAEConfig(
        input_length=INPUT_LENGTH, in_channels=1, dims=(64, 128, 256, 512),
        depths=(2, 2, 4, 2), latent_channels=16, latent_length=75,
        attn_heads=8, attn_dropout=0.0, global_latent_dim=128, trend_poly=2
    )
    ecg_ae = ECGAutoencoder(ecg_cfg).to(DEVICE)
    ecg_ae.load_state_dict(torch.load(PHASE1_ECG_PATH, map_location=DEVICE).get("model_state_dict", torch.load(PHASE1_ECG_PATH, map_location=DEVICE)))
    ecg_ae.eval()
    for p in ecg_ae.parameters(): p.requires_grad = False

    # 2. Load PPG2ECG Alignment Model (Lấy z_ppg)
    ppg_cfg = PPG2ECGConfig(
        input_length=INPUT_LENGTH, ppg_in_channels=1, dims=(64, 128, 256, 512),
        depths=(2, 2, 4, 2), latent_channels=16, latent_length=75,
        attn_heads=8, attn_dropout=0.0, use_derivatives=True, proj_dim=128
    )
    ppg_model = PPG2ECGModel(ecg_ae=ecg_ae, cfg=ppg_cfg).to(DEVICE)
    ppg_model.load_state_dict(torch.load(PHASE1_PPG_PATH, map_location=DEVICE).get("model_state_dict", torch.load(PHASE1_PPG_PATH, map_location=DEVICE)))
    ppg_model.eval()
    for p in ppg_model.parameters(): p.requires_grad = False

    # 3. Load Stage 2 Latent Rectified Flow
    flow_model = LatentRectifiedFlow(latent_channels=16, cond_channels=16, hidden_dim=128, num_blocks=6).to(DEVICE)
    flow_model.load_state_dict(torch.load(PHASE2_FLOW_PATH, map_location=DEVICE))
    flow_model.eval()
    for p in flow_model.parameters(): p.requires_grad = False
    
    return ecg_ae, ppg_model, flow_model

def euler_solve(flow_model, z_ppg, num_steps=10):
    """ Hàm giải ODE sinh predicted_ecg_latent từ z_ppg """
    B, C, L = z_ppg.shape
    # xt = torch.randn((B, C, L), device=DEVICE) # x_0 ~ N(0, I)
    xt = z_ppg.clone() 
    dt = 1.0 / num_steps
    
    for step in range(num_steps):
        t_val = step * dt
        t_tensor = torch.full((B,), t_val, device=DEVICE)
        v_pred = flow_model(xt, t_tensor, z_ppg)
        xt = xt + v_pred * dt
        
    return xt # Chính là predicted_ecg_latent (z_ecg_hat)

def run_stage2_testing():
    set_seed(SEED)
    
    ecg_ae, ppg_model, flow_model = load_models()

    print(f"[*] Loading Data from {TEST_DATA_PATH}...")
    dataset = LoadData(TEST_DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    gap = nn.AdaptiveAvgPool1d(1) 
    samples_collected = 0
    
    features = {
        'ecg_latent': [],
        'predicted_ecg_latent': []
    }
    
    # Lưu lại một vài sample thô để vẽ biểu đồ time-series
    plot_ecg_gt = []
    plot_ecg_recon = []

    print(f"[*] Extracting and Generating Latents for {NUM_SAMPLES_TO_VISUALIZE} samples...")
    for ecg, ppg, _ in tqdm(dataloader, desc="Inference Pass"):
        ecg = ecg.float().unsqueeze(1).to(DEVICE) if ecg.dim() == 2 else ecg.float().to(DEVICE)
        ppg = ppg.float().unsqueeze(1).to(DEVICE) if ppg.dim() == 2 else ppg.float().to(DEVICE)

        # 1. Trích xuất Ground Truth ECG Latent
        feat_ecg = ecg_ae.encoder(ecg)
        z_ecg = ecg_ae.latent_head(feat_ecg)

        
        # 2. Trích xuất PPG Latent (Condition)
        feat_ppg = ppg_model.ppg_encoder(ppg)
        z_ppg = ppg_model.ppg_latent_head(feat_ppg)
        
        # 3. Sinh Predicted ECG Latent bằng Rectified Flow
        z_ecg_hat = euler_solve(flow_model, z_ppg, num_steps=ODE_STEPS)
        
        # 4. Tái tạo tín hiệu ECG từ Predicted Latent
        reconstructed_ecg = ecg_ae.decoder(z_ecg_hat)

        # Lưu trữ cho t-SNE và Gaussian (Dùng Global Average Pooling để đưa về 1D)
        features['ecg_latent'].append(gap(z_ecg).squeeze(-1).cpu().numpy())
        features['predicted_ecg_latent'].append(gap(z_ecg_hat).squeeze(-1).cpu().numpy())
        
        # Lưu trữ một vài tín hiệu để vẽ time-series (lưu 5 sample đầu tiên)
        if len(plot_ecg_gt) < 5:
            plot_ecg_gt.append(ecg[0].squeeze().cpu().numpy())
            plot_ecg_recon.append(reconstructed_ecg[0].squeeze().cpu().numpy())

        samples_collected += ppg.shape[0]
        if samples_collected >= NUM_SAMPLES_TO_VISUALIZE:
            break

    # Gộp dữ liệu
    z_ecg_np = np.concatenate(features['ecg_latent'], axis=0)[:NUM_SAMPLES_TO_VISUALIZE]
    z_ecg_hat_np = np.concatenate(features['predicted_ecg_latent'], axis=0)[:NUM_SAMPLES_TO_VISUALIZE]

    # =========================================================
    # VISUALIZATION 1: ECG RECONSTRUCTION (TIME-SERIES)
    # =========================================================
    print("[*] Plotting Reconstructed ECG vs Ground Truth ECG...")
    fig, axes = plt.subplots(5, 1, figsize=(12, 12))
    for i in range(5):
        axes[i].plot(plot_ecg_gt[i], color='blue', label='Ground Truth ECG', alpha=0.7)
        axes[i].plot(plot_ecg_recon[i], color='red', label='Reconstructed ECG (from Flow)', alpha=0.7)
        axes[i].set_title(f"Sample {i+1}")
        axes[i].grid(True, alpha=0.3)
        if i == 0:
            axes[i].legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("stage2_ecg_reconstruction_timeseries.png", dpi=300)
    plt.close()

    # =========================================================
    # VISUALIZATION 2 & 3: T-SNE 2D VÀ 3D
    # =========================================================
    print("[*] Running t-SNE for 2D and 3D Visualization...")
    features_combined = np.vstack((z_ecg_hat_np, z_ecg_np))
    actual_samples = z_ecg_hat_np.shape[0]

    # Tính t-SNE 2D
    tsne_2d = TSNE(n_components=2, perplexity=30, random_state=SEED, init='pca', learning_rate=200.0)
    embeddings_2d = tsne_2d.fit_transform(features_combined)
    hat_2d = embeddings_2d[:actual_samples, :]
    gt_2d = embeddings_2d[actual_samples:, :]

    fig_2d, ax_2d = plt.subplots(figsize=(8, 6))
    ax_2d.scatter(gt_2d[:, 0], gt_2d[:, 1], c='blue', alpha=0.5, label='GT ECG Latent', s=15, edgecolors='none')
    ax_2d.scatter(hat_2d[:, 0], hat_2d[:, 1], c='red', alpha=0.5, label='Predicted ECG Latent', s=15, edgecolors='none')
    ax_2d.set_title('2D t-SNE: Predicted vs GT ECG Latent', fontsize=14, fontweight='bold')
    ax_2d.legend()
    ax_2d.grid(True, alpha=0.3)
    plt.savefig("stage2_tsne_2d.png", dpi=300)
    plt.close()

    # Tính t-SNE 3D
    tsne_3d = TSNE(n_components=3, perplexity=30, random_state=SEED, init='pca', learning_rate=200.0)
    embeddings_3d = tsne_3d.fit_transform(features_combined)
    hat_3d = embeddings_3d[:actual_samples, :]
    gt_3d = embeddings_3d[actual_samples:, :]

    fig_3d = plt.figure(figsize=(10, 8))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    ax_3d.scatter(gt_3d[:, 0], gt_3d[:, 1], gt_3d[:, 2], c='blue', alpha=0.6, label='GT ECG Latent', s=20, edgecolors='none')
    ax_3d.scatter(hat_3d[:, 0], hat_3d[:, 1], hat_3d[:, 2], c='red', alpha=0.6, label='Predicted ECG Latent', s=20, edgecolors='none')
    ax_3d.set_title('3D t-SNE: Predicted vs GT ECG Latent', fontsize=16, fontweight='bold')
    ax_3d.legend()
    ax_3d.view_init(elev=20, azim=45) 
    plt.savefig("stage2_tsne_3d.png", dpi=300)
    plt.close()

    # =========================================================
    # VISUALIZATION 4: VẼ ĐƯỜNG CONG GAUSSIAN 1D (DENSITY)
    # =========================================================
    print("[*] Running Theoretical Gaussian Curve...")
    z_ecg_flat = z_ecg_np.flatten()
    z_hat_flat = z_ecg_hat_np.flatten()

    mu_gt, std_gt = z_ecg_flat.mean(), z_ecg_flat.std()
    mu_hat, std_hat = z_hat_flat.mean(), z_hat_flat.std()

    plt.figure(figsize=(10, 5))
    x_min_plot = min(mu_gt, mu_hat) - 4 * max(std_gt, std_hat)
    x_max_plot = max(mu_gt, mu_hat) + 4 * max(std_gt, std_hat)
    x = np.linspace(x_min_plot, x_max_plot, 500)

    plt.plot(x, stats.norm.pdf(x, mu_gt, std_gt), color='blue', lw=2, 
             label=f'GT ECG (mean={mu_gt:.4f}, var={std_gt**2:.4f})')
    plt.plot(x, stats.norm.pdf(x, mu_hat, std_hat), color='red', lw=2, ls='--', 
             label=f'Predicted ECG (mean={mu_hat:.4f}, var={std_hat**2:.4f})')

    plt.axvline(mu_gt, color='blue', alpha=0.5, ls=':')
    plt.axvline(mu_hat, color='red', alpha=0.5, ls=':')

    plt.title("Gaussian Distribution: Predicted ECG Latent vs Ground Truth", fontsize=14)
    plt.xlabel("Latent value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("stage2_gaussian_1d.png", dpi=300)
    plt.close()

    # =========================================================
    # VISUALIZATION 5: VẼ MẶT CONG GAUSSIAN 3D
    # =========================================================
    print("[*] Running Theoretical 3D Gaussian Surface...")
    fig_theo_3d = plt.figure(figsize=(12, 8))
    ax_theo_3d = fig_theo_3d.add_subplot(111, projection='3d')

    shift = mu_hat - mu_gt
    x_range = np.linspace(-3*std_gt - abs(shift), 3*std_gt + abs(shift), 100)
    y_range = np.linspace(-3*std_gt - abs(shift), 3*std_gt + abs(shift), 100)
    X_theo, Y_theo = np.meshgrid(x_range, y_range)
    
    pos_theo = np.empty(X_theo.shape + (2,))
    pos_theo[:, :, 0] = X_theo
    pos_theo[:, :, 1] = Y_theo

    rv_gt = stats.multivariate_normal([0, 0], [[std_gt**2, 0], [0, std_gt**2]])
    rv_hat = stats.multivariate_normal([shift, shift], [[std_hat**2, 0], [0, std_hat**2]])

    ax_theo_3d.plot_surface(X_theo, Y_theo, rv_gt.pdf(pos_theo), cmap='Blues', alpha=0.7, antialiased=True)
    ax_theo_3d.plot_surface(X_theo, Y_theo, rv_hat.pdf(pos_theo), cmap='Oranges', alpha=0.6, antialiased=True)

    ax_theo_3d.set_title("3D Theoretical Latent Distribution (GT vs Predicted)", fontsize=16, fontweight='bold')
    ax_theo_3d.set_zticks([]) 

    legend_elements_theo = [
        Patch(facecolor='blue', alpha=0.6, label='GT ECG Distribution'),
        Patch(facecolor='orange', alpha=0.6, label='Predicted ECG Distribution')
    ]
    ax_theo_3d.legend(handles=legend_elements_theo, loc='lower right', fontsize=12)
    ax_theo_3d.view_init(elev=30, azim=-45)

    plt.savefig("stage2_gaussian_3d_surface.png", dpi=300)
    print("[*] All tests and visualizations completed successfully!")
    plt.close()

if __name__ == "__main__":
    run_stage2_testing()