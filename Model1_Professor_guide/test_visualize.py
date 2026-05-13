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

# Thêm thư viện hỗ trợ vẽ 3D và tính toán phân phối Gaussian KDE
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde
from matplotlib.patches import Patch

from load_data import LoadData 
from CLIP_2stage import ECG_PPG_Fusion_Model

# ==========================================
# CẤU HÌNH (CONFIGURATIONS)
# ==========================================
SEED = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
INPUT_LENGTH = 2400
OUTPUT_EMBED_DIM = 128
TEST_DATA_PATH = '/home/linhhima/Diffusion datasets/test.npz'
CLIP_MODEL_PATH = "/home/linhhima/PPG_ECG/best_phase2_fusion.pth" 
NUM_SAMPLES_TO_VISUALIZE = 1900 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_tsne_layer_evolution():
    set_seed(SEED)
    
    print(f"1. Loading Model from {CLIP_MODEL_PATH}...")
    model = ECG_PPG_Fusion_Model(embed_dim=OUTPUT_EMBED_DIM, freq_dim=256, target_length=INPUT_LENGTH).to(DEVICE)
    
    if torch.cuda.is_available():
        checkpoint = torch.load(CLIP_MODEL_PATH)
    else:
        checkpoint = torch.load(CLIP_MODEL_PATH, map_location='cpu')
        
    model.load_state_dict(checkpoint)
    model.eval()

    print(f"2. Loading Data from {TEST_DATA_PATH}...")
    try:
        dataset = LoadData(TEST_DATA_PATH)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    features = {
        'Layer 1': {'ppg': [], 'ecg': []},
        'Layer 2': {'ppg': [], 'ecg': []},
        'Layer 3': {'ppg': [], 'ecg': []},
        'Layer 4': {'ppg': [], 'ecg': []},
        'Final Latent': {'ppg': [], 'ecg': []}
    }
    
    gap = nn.AdaptiveAvgPool1d(1) 
    samples_collected = 0
    data_warning_triggered = False

    print(f"3. Extracting Features for up to {NUM_SAMPLES_TO_VISUALIZE} samples...")
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Forward Pass"):
            ecg = batch_data[0]
            ppg = batch_data[1]
            
            if not data_warning_triggered and torch.allclose(ecg, ppg, atol=1e-5):
                warnings.warn("CẢNH BÁO: Tín hiệu PPG và ECG từ DataLoader hoàn toàn giống nhau! Hãy kiểm tra lại file load_data.py")
                data_warning_triggered = True

            ecg_target = ecg.float().unsqueeze(1).to(DEVICE)
            ppg_input = ppg.float().unsqueeze(1).to(DEVICE)

            ecg_f4, ecg_f_list = model.encode_ecg.time_branch(ecg_target)
            ppg_f4, ppg_f_list = model.encode_ppg.time_branch(ppg_input)
            
            z_ecg_1d, _ = model.encode_ecg(ecg_target)
            z_ppg_1d, _ = model.encode_ppg(ppg_input)

            features['Layer 1']['ecg'].append(gap(ecg_f_list[0]).squeeze(-1).cpu().numpy())
            features['Layer 1']['ppg'].append(gap(ppg_f_list[0]).squeeze(-1).cpu().numpy())
            
            features['Layer 2']['ecg'].append(gap(ecg_f_list[1]).squeeze(-1).cpu().numpy())
            features['Layer 2']['ppg'].append(gap(ppg_f_list[1]).squeeze(-1).cpu().numpy())
            
            features['Layer 3']['ecg'].append(gap(ecg_f_list[2]).squeeze(-1).cpu().numpy())
            features['Layer 3']['ppg'].append(gap(ppg_f_list[2]).squeeze(-1).cpu().numpy())
            
            features['Layer 4']['ecg'].append(gap(ecg_f4).squeeze(-1).cpu().numpy())
            features['Layer 4']['ppg'].append(gap(ppg_f4).squeeze(-1).cpu().numpy())
            
            features['Final Latent']['ecg'].append(z_ecg_1d.cpu().numpy())
            features['Final Latent']['ppg'].append(z_ppg_1d.cpu().numpy())

            samples_collected += ppg_input.shape[0]
            if samples_collected >= NUM_SAMPLES_TO_VISUALIZE:
                break
    
    # Biến tạm để lưu tọa độ 2D của Final Latent cho bước 6
    final_ppg_2d = None
    final_ecg_2d = None

    # ---------------------------------------------------------
    # BƯỚC 4: VẼ 2D t-SNE CHO TẤT CẢ CÁC LAYER
    # ---------------------------------------------------------
    print("4. Running 2D t-SNE and plotting layers...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    stage_names = ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Final Latent']
    
    for idx, stage in enumerate(stage_names):
        z_ppg_np = np.concatenate(features[stage]['ppg'], axis=0)[:NUM_SAMPLES_TO_VISUALIZE]
        z_ecg_np = np.concatenate(features[stage]['ecg'], axis=0)[:NUM_SAMPLES_TO_VISUALIZE]
        actual_samples = z_ppg_np.shape[0]

        features_combined = np.vstack((z_ppg_np, z_ecg_np))
        
        tsne_2d = TSNE(n_components=2, perplexity=30, random_state=SEED, init='pca', learning_rate=200.0)
        embeddings_2d = tsne_2d.fit_transform(features_combined)

        ppg_2d = embeddings_2d[:actual_samples, :]
        ecg_2d = embeddings_2d[actual_samples:, :]

        # Lưu lại toạ độ 2D của Final Latent để dùng vẽ phân phối Gaussian ở Bước 6
        if stage == 'Final Latent':
            final_ppg_2d = ppg_2d.copy()
            final_ecg_2d = ecg_2d.copy()

        ax = axes[idx]
        ax.scatter(ecg_2d[:, 0], ecg_2d[:, 1], c='blue', alpha=0.5, label='ECG (Target)', s=15, edgecolors='none')
        ax.scatter(ppg_2d[:, 0], ppg_2d[:, 1], c='red', alpha=0.5, label='PPG (Source)', s=15, edgecolors='none')
        
        ax.set_title(f'{stage} Alignment', fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=10, markerscale=2)
            
        del features_combined

    axes[5].axis('off')
    plt.suptitle('Evolution of PPG and ECG Feature Alignment across ResNet-50 Layers', fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    
    output_path_2d = "tsne_layer_evolution_2d.png"
    plt.savefig(output_path_2d, dpi=300)
    print(f"Done! 2D Visualization saved to {output_path_2d}")
    plt.close() 

    # ---------------------------------------------------------
    # BƯỚC 5: VẼ 3D t-SNE (SCATTER PLOT ĐIỂM) CHO LỚP FINAL LATENT
    # ---------------------------------------------------------
    print("\n5. Running 3D t-SNE Scatter for Final Latent Space...")
    z_ppg_final = np.concatenate(features['Final Latent']['ppg'], axis=0)[:NUM_SAMPLES_TO_VISUALIZE]
    z_ecg_final = np.concatenate(features['Final Latent']['ecg'], axis=0)[:NUM_SAMPLES_TO_VISUALIZE]
    actual_samples_3d = z_ppg_final.shape[0]

    features_combined_3d = np.vstack((z_ppg_final, z_ecg_final))
    tsne_3d = TSNE(n_components=3, perplexity=30, random_state=SEED, init='pca', learning_rate=200.0)
    embeddings_3d = tsne_3d.fit_transform(features_combined_3d)

    ppg_3d = embeddings_3d[:actual_samples_3d, :]
    ecg_3d = embeddings_3d[actual_samples_3d:, :]

    fig_3d = plt.figure(figsize=(10, 8))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    ax_3d.scatter(ecg_3d[:, 0], ecg_3d[:, 1], ecg_3d[:, 2], c='blue', alpha=0.6, label='ECG (Target)', s=20, edgecolors='none')
    ax_3d.scatter(ppg_3d[:, 0], ppg_3d[:, 1], ppg_3d[:, 2], c='red', alpha=0.6, label='PPG (Source)', s=20, edgecolors='none')

    ax_3d.set_title('3D t-SNE Visualization of Final Latent Space', fontsize=16, fontweight='bold')
    ax_3d.set_xlabel('t-SNE 1')
    ax_3d.set_ylabel('t-SNE 2')
    ax_3d.set_zlabel('t-SNE 3')
    ax_3d.legend(fontsize=12, markerscale=2)
    ax_3d.view_init(elev=20, azim=45) 

    output_path_3d = "tsne_final_latent_3d.png"
    plt.savefig(output_path_3d, dpi=300)
    print(f"Done! 3D Scatter Visualization saved to {output_path_3d}")
    plt.close()

    # ---------------------------------------------------------
    # BƯỚC 6: VẼ 3D GAUSSIAN HISTOGRAM (DENSITY SURFACE)
    # ---------------------------------------------------------
    print("\n6. Running 3D Gaussian Density Histogram for Final Latent Space...")
    
    fig_kde = plt.figure(figsize=(12, 10))
    ax_kde = fig_kde.add_subplot(111, projection='3d')

    # Lấy giới hạn trục x, y từ tọa độ 2D của Bước 4 để tạo lưới
    x_min = min(final_ppg_2d[:, 0].min(), final_ecg_2d[:, 0].min()) - 5
    x_max = max(final_ppg_2d[:, 0].max(), final_ecg_2d[:, 0].max()) + 5
    y_min = min(final_ppg_2d[:, 1].min(), final_ecg_2d[:, 1].min()) - 5
    y_max = max(final_ppg_2d[:, 1].max(), final_ecg_2d[:, 1].max()) + 5

    # Tạo lưới 2D (100x100) để tính mật độ
    X, Y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])

    # Tính KDE (Kernel Density Estimation) cho PPG
    values_ppg = np.vstack([final_ppg_2d[:, 0], final_ppg_2d[:, 1]])
    kernel_ppg = gaussian_kde(values_ppg)
    Z_ppg = np.reshape(kernel_ppg(positions).T, X.shape)

    # Tính KDE cho ECG
    values_ecg = np.vstack([final_ecg_2d[:, 0], final_ecg_2d[:, 1]])
    kernel_ecg = gaussian_kde(values_ecg)
    Z_ecg = np.reshape(kernel_ecg(positions).T, X.shape)

    # Vẽ phân phối dạng đồi núi 3D (Surface Plot)
    # Sử dụng cmap Blues cho ECG và Reds cho PPG
    ax_kde.plot_surface(X, Y, Z_ecg, cmap='Blues', alpha=0.6, edgecolor='none')
    ax_kde.plot_surface(X, Y, Z_ppg, cmap='Reds', alpha=0.5, edgecolor='none')

    ax_kde.set_title('3D Gaussian Density Distribution (Final Latent)', fontsize=16, fontweight='bold')
    ax_kde.set_xlabel('t-SNE Dimension 1')
    ax_kde.set_ylabel('t-SNE Dimension 2')
    ax_kde.set_zlabel('Density (Probability)')

    # Cấu hình chú thích (Legend) do plot_surface không hỗ trợ label trực tiếp
    legend_elements = [
        Patch(facecolor='blue', alpha=0.6, label='ECG (Target) Density'),
        Patch(facecolor='red', alpha=0.5, label='PPG (Source) Density')
    ]
    ax_kde.legend(handles=legend_elements, loc='upper right', fontsize=12)

    # Chỉnh góc nhìn
    ax_kde.view_init(elev=35, azim=45)
    
    output_path_kde = "tsne_final_latent_gaussian_density.png"
    plt.savefig(output_path_kde, dpi=300)
    print(f"Done! 3D Gaussian Density Visualization saved to {output_path_kde}")
    plt.close()

if __name__ == "__main__":
    run_tsne_layer_evolution()