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

# Libraries for 3D visualization and Gaussian KDE/Theoretical Distribution calculations
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde
import scipy.stats as stats
from matplotlib.patches import Patch

from load_data import LoadData 
from ecg2ecg import ECGAutoencoder, ECGAEConfig
from ppg2ecg import PPG2ECGModel, PPG2ECGConfig 

# ==========================================
# CONFIGURATIONS
# ==========================================
SEED = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
INPUT_LENGTH = 2400
TEST_DATA_PATH = '/home/linhhima/Diffusion datasets/combined_segment_split_test.npz'

# Model weight paths
PHASE1_MODEL_PATH = '/home/linhhima/PPG_ECG/PPG2ECG_RF/Proposed/saved_models_ecg_vae/best_ecg_autoencoder.pth'
PHASE2_MODEL_PATH = '/home/linhhima/PPG_ECG/PPG2ECG_RF/Proposed/saved_models_alignment/best_ppg_alignment.pth'

NUM_SAMPLES_TO_VISUALIZE = 1900

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_model():
    print(f"[*] Loading PPG2ECG Alignment Model on {DEVICE}...")
    
    ecg_cfg = ECGAEConfig(
        input_length=INPUT_LENGTH, in_channels=1, dims=(64, 128, 256, 512),
        depths=(2, 2, 4, 2), latent_channels=16, latent_length=75,
        attn_heads=8, attn_dropout=0.0, global_latent_dim=128, trend_poly=2
    )

    ecg_ae = ECGAutoencoder(ecg_cfg).to(DEVICE)
    if not os.path.exists(PHASE1_MODEL_PATH):
        raise FileNotFoundError(f"Error: Phase 1 weights file required: {PHASE1_MODEL_PATH}")
    ckpt1 = torch.load(PHASE1_MODEL_PATH, map_location=DEVICE)
    ecg_ae.load_state_dict(ckpt1.get("model_state_dict", ckpt1))
    
    ppg_cfg = PPG2ECGConfig(
        input_length=INPUT_LENGTH, ppg_in_channels=1, dims=(64, 128, 256, 512),
        depths=(2, 2, 4, 2), latent_channels=16, latent_length=75,
        attn_heads=8, attn_dropout=0.0, use_derivatives=True, proj_dim=128
    )
    
    model = PPG2ECGModel(ecg_ae=ecg_ae, cfg=ppg_cfg).to(DEVICE)
    if not os.path.exists(PHASE2_MODEL_PATH):
        raise FileNotFoundError(f"Error: Phase 2 weights file not found: {PHASE2_MODEL_PATH}")
    
    checkpoint = torch.load(PHASE2_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()
   
    return model

def run_tsne_layer_evolution():
    set_seed(SEED)
    
    model = load_model()

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
                warnings.warn("WARNING: PPG and ECG signals are completely identical!")
                data_warning_triggered = True

            ecg_target = ecg.float().unsqueeze(1).to(DEVICE) if ecg.dim() == 2 else ecg.float().to(DEVICE)
            ppg_input = ppg.float().unsqueeze(1).to(DEVICE) if ppg.dim() == 2 else ppg.float().to(DEVICE)

            # Extract ECG Layers
            e = model.ecg_encoder.stem(ecg_target)
            e1 = model.ecg_encoder.stage1(e)
            e = model.ecg_encoder.down1(e1)
            e2 = model.ecg_encoder.stage2(e)
            e = model.ecg_encoder.down2(e2)
            e3 = model.ecg_encoder.stage3(e)
            e = model.ecg_encoder.down3(e3)
            e4 = model.ecg_encoder.stage4(e)
            feat_ecg = model.ecg_encoder.bottleneck_attn(e4)
            z_ecg = model.ecg_latent_head(feat_ecg)
            z_ecg_pool = model.pool_temporal(z_ecg)

            # Extract PPG Layers
            p = model.ppg_encoder.prep(ppg_input)
            p = model.ppg_encoder.stem(p)
            p = model.ppg_encoder.patchify(p)
            p1 = model.ppg_encoder.stage1(p)
            p = model.ppg_encoder.down1(p1)
            p2 = model.ppg_encoder.stage2(p)
            p = model.ppg_encoder.down2(p2)
            p3 = model.ppg_encoder.stage3(p)
            p = model.ppg_encoder.down3(p3)
            p4 = model.ppg_encoder.stage4(p)
            feat_ppg = model.ppg_encoder.bottleneck_attn(p4)
            z_ppg = model.ppg_latent_head(feat_ppg)
            z_ppg_pool = model.pool_temporal(z_ppg)

            features['Layer 1']['ecg'].append(gap(e1).squeeze(-1).cpu().numpy())
            features['Layer 1']['ppg'].append(gap(p1).squeeze(-1).cpu().numpy())
            features['Layer 2']['ecg'].append(gap(e2).squeeze(-1).cpu().numpy())
            features['Layer 2']['ppg'].append(gap(p2).squeeze(-1).cpu().numpy())
            features['Layer 3']['ecg'].append(gap(e3).squeeze(-1).cpu().numpy())
            features['Layer 3']['ppg'].append(gap(p3).squeeze(-1).cpu().numpy())
            features['Layer 4']['ecg'].append(gap(e4).squeeze(-1).cpu().numpy())
            features['Layer 4']['ppg'].append(gap(p4).squeeze(-1).cpu().numpy())
            features['Final Latent']['ecg'].append(z_ecg_pool.cpu().numpy())
            features['Final Latent']['ppg'].append(z_ppg_pool.cpu().numpy())

            samples_collected += ppg_input.shape[0]
            if samples_collected >= NUM_SAMPLES_TO_VISUALIZE:
                break
    
    final_ppg_2d = None
    final_ecg_2d = None

    # ---------------------------------------------------------
    # STEP 4: PLOT 2D t-SNE FOR ALL ENCODER LAYERS
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
    plt.suptitle('Evolution of PPG and ECG Feature Alignment across Encoder Stages', fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.savefig("tsne_layer_evolution_2d.png", dpi=300)
    plt.close() 

    # ---------------------------------------------------------
    # STEP 5: PLOT 3D t-SNE SCATTER
    # ---------------------------------------------------------
    print("\n5. Running 3D t-SNE Scatter for Final Latent Space...")
    z_ppg_final = np.concatenate(features['Final Latent']['ppg'], axis=0)[:NUM_SAMPLES_TO_VISUALIZE]
    z_ecg_final = np.concatenate(features['Final Latent']['ecg'], axis=0)[:NUM_SAMPLES_TO_VISUALIZE]
    
    features_combined_3d = np.vstack((z_ppg_final, z_ecg_final))
    tsne_3d = TSNE(n_components=3, perplexity=30, random_state=SEED, init='pca', learning_rate=200.0)
    embeddings_3d = tsne_3d.fit_transform(features_combined_3d)

    ppg_3d = embeddings_3d[:actual_samples, :]
    ecg_3d = embeddings_3d[actual_samples:, :]

    fig_3d = plt.figure(figsize=(10, 8))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    ax_3d.scatter(ecg_3d[:, 0], ecg_3d[:, 1], ecg_3d[:, 2], c='blue', alpha=0.6, label='ECG (Target)', s=20, edgecolors='none')
    ax_3d.scatter(ppg_3d[:, 0], ppg_3d[:, 1], ppg_3d[:, 2], c='red', alpha=0.6, label='PPG (Source)', s=20, edgecolors='none')
    ax_3d.set_title('3D t-SNE Visualization of Final Latent Space', fontsize=16, fontweight='bold')
    ax_3d.legend(fontsize=12, markerscale=2)
    ax_3d.view_init(elev=20, azim=45) 
    plt.savefig("tsne_final_latent_3d.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # STEPS 6 & 7 (LEGACY): t-SNE KDE DENSITY (Optional, commented out)
    # ---------------------------------------------------------
    # These steps compute density estimates over distorted t-SNE coordinates.
    # Replaced by standard Gaussian visualizations in Steps 8 & 9 below.

    # ---------------------------------------------------------
    # STEP 8: PLOT 1D THEORETICAL GAUSSIAN CURVE
    # ---------------------------------------------------------
    print("\n8. Running Theoretical Gaussian Curve for Final Latent Space...")
    # Calculate actual Mean and Std across the entire Latent tensor (flattened to 1D)
    z_ppg_flat = z_ppg_final.flatten()
    z_ecg_flat = z_ecg_final.flatten()

    mu_p, std_p = z_ppg_flat.mean(), z_ppg_flat.std()
    mu_e, std_e = z_ecg_flat.mean(), z_ecg_flat.std()

    plt.figure(figsize=(10, 5))
    
    # Define X-axis range to enclose both distributions
    x_min_plot = min(mu_p, mu_e) - 4 * max(std_p, std_e)
    x_max_plot = max(mu_p, mu_e) + 4 * max(std_p, std_e)
    x = np.linspace(x_min_plot, x_max_plot, 500)

    # Plot Probability Density Function (PDF)
    plt.plot(x, stats.norm.pdf(x, mu_p, std_p), color='red', lw=2, 
             label=f'z_ppg (Source) (mean={mu_p:.4f}, var={std_p**2:.4f})')
    plt.plot(x, stats.norm.pdf(x, mu_e, std_e), color='blue', lw=2, ls='--', 
             label=f'z_ecg (Ground Truth) (mean={mu_e:.4f}, var={std_e**2:.4f})')

    # Draw central vertical dotted mean lines (Mu)
    plt.axvline(mu_p, color='red', alpha=0.5, ls=':')
    plt.axvline(mu_e, color='blue', alpha=0.5, ls=':')

    plt.title("Gaussian Distribution: Source z_ppg vs Target z_ecg", fontsize=14)
    plt.xlabel("Latent value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("gaussian_theoretical_curve.png", dpi=300)
    print("Done! Theoretical Gaussian Curve saved to gaussian_theoretical_curve.png")
    plt.close()

    # ---------------------------------------------------------
    # STEP 9: PLOT 3D THEORETICAL GAUSSIAN SURFACE
    # ---------------------------------------------------------
    print("\n9. Running Theoretical 3D Gaussian Surface for Final Latent Space...")
    fig_theo_3d = plt.figure(figsize=(12, 8))
    ax_theo_3d = fig_theo_3d.add_subplot(111, projection='3d')

    # Actual mean shift between the two distributions
    shift = mu_p - mu_e

    # Grid mesh coordinates
    x_range = np.linspace(-3*std_e - abs(shift), 3*std_e + abs(shift), 100)
    y_range = np.linspace(-3*std_e - abs(shift), 3*std_e + abs(shift), 100)
    X_theo, Y_theo = np.meshgrid(x_range, y_range)
    
    pos_theo = np.empty(X_theo.shape + (2,))
    pos_theo[:, :, 0] = X_theo
    pos_theo[:, :, 1] = Y_theo

    # Initialize Multivariate Normal distributions
    # ECG centered at (0,0), PPG offset by `shift`
    rv_e = stats.multivariate_normal([0, 0], [[std_e**2, 0], [0, std_e**2]])
    rv_p = stats.multivariate_normal([shift, shift], [[std_p**2, 0], [0, std_p**2]])

    # Render overlapping surfaces
    ax_theo_3d.plot_surface(X_theo, Y_theo, rv_e.pdf(pos_theo), cmap='Blues', alpha=0.7, antialiased=True)
    ax_theo_3d.plot_surface(X_theo, Y_theo, rv_p.pdf(pos_theo), cmap='Oranges', alpha=0.6, antialiased=True)

    ax_theo_3d.set_title("3D Theoretical Latent Distribution Alignment", fontsize=16, fontweight='bold')
    
    # Hide Z-axis ticks for cleaner illustration
    ax_theo_3d.set_zticks([]) 

    # Add legend handles
    legend_elements_theo = [
        Patch(facecolor='blue', alpha=0.6, label='ECG Distribution'),
        Patch(facecolor='orange', alpha=0.6, label='PPG Distribution')
    ]
    ax_theo_3d.legend(handles=legend_elements_theo, loc='lower right', fontsize=12)
    
    # Adjust camera angle (Elevation = 30, Azimuth = -45)
    ax_theo_3d.view_init(elev=30, azim=-45)

    plt.savefig("gaussian_theoretical_3d_surface.png", dpi=300)
    print("Done! 3D Theoretical Gaussian Surface saved to gaussian_theoretical_3d_surface.png")
    plt.close()

if __name__ == "__main__":
    run_tsne_layer_evolution()