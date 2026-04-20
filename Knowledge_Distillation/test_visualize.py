import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE  
import torch.nn.functional as F   
from load_data import LoadData
from CLIP import ECGEssembleCLIP  
import random

# --- CONFIGURATION ---
SEED = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
INPUT_LENGTH = 2400
OUTPUT_EMBED_DIM = 128
TEST_DATA_PATH = 'processed_data/mimic3_v1_2400_train.npz'
CLIP_MODEL_PATH = "multitask_best_model_best.pth" 
SAMPLING_RATE = 125
NUM_SAMPLES_TO_VISUALIZE = 2000 
LAYERS_TO_VISUALIZE = ['layer1', 'layer2', 'layer3', 'layer4']

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_model():
    print(f"Loading model on {DEVICE}...")
    model = ECGEssembleCLIP(embed_dim=OUTPUT_EMBED_DIM).to(DEVICE)

    if torch.cuda.is_available():
        weights = torch.load(CLIP_MODEL_PATH)
    else:
        weights = torch.load(CLIP_MODEL_PATH, map_location='cpu')

    model.load_state_dict(weights)
    model.eval()
    
    return model

def visualize_feature_space():
    set_seed(SEED)
    model = load_model()
    
    try:
        test_dataset = LoadData(TEST_DATA_PATH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    except Exception as e:
        print(f"Error loading data for t-SNE: {e}")
        return

    activations = {layer: {'ppg': [], 'ecg': []} for layer in LAYERS_TO_VISUALIZE}

    def get_activation(layer_name, modality):
        def hook(module, input, output):
            out_tensor = output[0] if isinstance(output, tuple) else output
            if len(out_tensor.shape) == 3:
                out_tensor = out_tensor.mean(dim=2) 
            activations[layer_name][modality].append(out_tensor.detach().cpu().numpy())
        return hook

    try:
        for layer_name in LAYERS_TO_VISUALIZE:
            getattr(model.encode_ppg.time_branch, layer_name).register_forward_hook(get_activation(layer_name, 'ppg'))
            getattr(model.encode_ecg.time_branch, layer_name).register_forward_hook(get_activation(layer_name, 'ecg'))

        print("Hooks registered successfully on ResNet1D layers.")
    except AttributeError as e:
        print(f"Error setting hooks: {e}")
        return

    print(f"Extracting features for max {NUM_SAMPLES_TO_VISUALIZE} samples...")
    collected_samples = 0
    
    final_ppg_embeddings = []
    final_ecg_embeddings = []
    
    fused_3d_ppg_list = []
    fused_3d_ecg_list = []

    sum_ecg_fft = None
    sum_ppg_fft = None

    with torch.no_grad():
        for ecg, ppg, _ in test_loader:
            if collected_samples >= NUM_SAMPLES_TO_VISUALIZE:
                break
                
            ecg_input = ecg.to(DEVICE).float().unsqueeze(1)
            ppg_input = ppg.to(DEVICE).float().unsqueeze(1)
            
            # Only save the FFT sum to calculate the average, saving RAM instead of storing each sample
            batch_ecg_fft = torch.abs(torch.fft.rfft(ecg_input.squeeze(1))).sum(dim=0).cpu().numpy()
            batch_ppg_fft = torch.abs(torch.fft.rfft(ppg_input.squeeze(1))).sum(dim=0).cpu().numpy()
            
            if sum_ecg_fft is None:
                sum_ecg_fft = batch_ecg_fft
                sum_ppg_fft = batch_ppg_fft
            else:
                sum_ecg_fft += batch_ecg_fft
                sum_ppg_fft += batch_ppg_fft

            # Extracting fused_3d
            _, ecg_fused_3d, _ = model.encode_ecg(ecg_input)
            _, ppg_fused_3d, _ = model.encode_ppg(ppg_input)

            # Global Average Pooling to get 2D for t-SNE
            ecg_f3d_pooled = ecg_fused_3d.mean(dim=2).cpu().numpy()
            ppg_f3d_pooled = ppg_fused_3d.mean(dim=2).cpu().numpy()
            
            fused_3d_ecg_list.append(ecg_f3d_pooled)
            fused_3d_ppg_list.append(ppg_f3d_pooled)

            # Extracting final embedding
            ecg_embed, ppg_embed, _, _, _= model(ecg_input, ppg_input)
            
            ecg_embed = F.normalize(ecg_embed, p=2, dim=-1)
            ppg_embed = F.normalize(ppg_embed, p=2, dim=-1)

            final_ecg_embeddings.append(ecg_embed.cpu().numpy())
            final_ppg_embeddings.append(ppg_embed.cpu().numpy())
            
            collected_samples += ppg_input.size(0)

    actual_samples = np.concatenate(final_ppg_embeddings, axis=0)[:NUM_SAMPLES_TO_VISUALIZE].shape[0]

    # ---------------------------------------------------------
    # 1. PLOT LAYERS 1 -> 4
    # ---------------------------------------------------------
    print("\n [1/4] Running t-SNE for 4 intermediate layers...")
    fig1, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()

    for idx, layer_name in enumerate(LAYERS_TO_VISUALIZE):
        ax = axes[idx]
        print(f"   -> Processing {layer_name}...")
        
        ppg_layer_feats = np.concatenate(activations[layer_name]['ppg'], axis=0)[:actual_samples]
        ecg_layer_feats = np.concatenate(activations[layer_name]['ecg'], axis=0)[:actual_samples]
        
        all_feats = np.vstack((ppg_layer_feats, ecg_layer_feats))
        
        tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=SEED)
        all_2d = tsne.fit_transform(all_feats)
        
        ppg_2d = all_2d[:actual_samples]
        ecg_2d = all_2d[actual_samples:]

        ax.scatter(ppg_2d[:, 0], ppg_2d[:, 1], color='tomato', label='PPG', alpha=0.7, s=40, edgecolors='w')
        ax.scatter(ecg_2d[:, 0], ecg_2d[:, 1], color='royalblue', label='ECG', alpha=0.7, s=40, edgecolors='w')

        for i in range(actual_samples):
            ax.plot([ppg_2d[i, 0], ecg_2d[i, 0]], [ppg_2d[i, 1], ecg_2d[i, 1]], 
                    color='gray', alpha=0.1, linewidth=0.5, zorder=0)

        ax.set_title(f"After {layer_name.upper()} (Dim: {ppg_layer_feats.shape[1]})", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)

    fig1.suptitle("Evolution of PPG and ECG Embeddings through Layers", fontsize=20, fontweight='bold', color='darkred')
    fig1.tight_layout(rect=[0, 0, 1, 0.95])

    # # ---------------------------------------------------------
    # # 2. FREQUENCY SPECTRUM (BEFORE AND AFTER FFT_MLP)
    # # ---------------------------------------------------------
    # print(f"\n [2/4] Plotting Average Frequency Spectrum (Before and After Filter)...")
    
    # # Calculate the average FFT spectrum
    # avg_ecg_fft = sum_ecg_fft / actual_samples
    # avg_ppg_fft = sum_ppg_fft / actual_samples
    
    # # Create X-axis (Frequency in Hz)
    # freqs_hz = np.fft.rfftfreq(INPUT_LENGTH, d=1.0/SAMPLING_RATE)
    
    # # Extract filter parameters from the Model (Both are High-pass filters now)
    # ecg_branch = model.encode_ecg.freq_branch
    # ppg_branch = model.encode_ppg.freq_branch
    
    # ecg_fc_norm = max(ecg_branch.fc.item(), 1e-3)
    # ppg_fc_norm = max(ppg_branch.fc.item(), 1e-3)
    
    # ecg_fc_hz = ecg_fc_norm * (SAMPLING_RATE / 2.0)
    # ppg_fc_hz = ppg_fc_norm * (SAMPLING_RATE / 2.0)
    
    # freqs_norm_ecg = np.maximum(ecg_branch.freqs.cpu().numpy(), 1e-5)
    # freqs_norm_ppg = np.maximum(ppg_branch.freqs.cpu().numpy(), 1e-5)
    
    # # Calculate the filter curve H (Both are HIGH-PASS Filters now)
    # # Notice: (fc / freqs) instead of (freqs / fc)
    # H_ecg = 1.0 / (1.0 + (ecg_fc_norm / freqs_norm_ecg) ** (2 * ecg_branch.order)) 
    # H_ppg = 1.0 / (1.0 + (ppg_fc_norm / freqs_norm_ppg) ** (2 * ppg_branch.order)) 
    
    # # Apply the filter to the average signal
    # current_len = avg_ecg_fft.shape[0]
    # H_ecg_applied = H_ecg[:current_len] if current_len < len(H_ecg) else H_ecg
    # H_ppg_applied = H_ppg[:current_len] if current_len < len(H_ppg) else H_ppg
    
    # filtered_ecg_fft = avg_ecg_fft * H_ecg_applied
    # filtered_ppg_fft = avg_ppg_fft * H_ppg_applied
    
    # # --- PLOT 2x2 SUBPLOTS (HISTOGRAM STYLE) ---
    # fig_spectrum, axes_spec = plt.subplots(2, 2, figsize=(16, 10))
    
    # # Tính chiều rộng của mỗi cột (khoảng cách giữa 2 bin tần số)
    # bar_width = freqs_hz[1] - freqs_hz[0]
    
    # # Get the maximum value of the original signal to fix the Y-axis
    # max_ecg = avg_ecg_fft.max() if avg_ecg_fft.max() > 0 else 1
    # max_ppg = avg_ppg_fft.max() if avg_ppg_fft.max() > 0 else 1
    
    # # Add 5% padding so the peaks don't touch the top of the graph
    # ylim_ecg = (0, max_ecg * 1.05)
    # ylim_ppg = (0, max_ppg * 1.05)
    
    # # [Top-Left] ECG: BEFORE (High-Pass Histogram)
    # ax = axes_spec[0, 0]
    # # Dùng ax.bar để vẽ histogram
    # ax.bar(freqs_hz, avg_ecg_fft, width=bar_width, color='gray', alpha=0.7, label='Original ECG Average')
    # # Giữ nguyên đường nối liền cho màng lọc H để dễ nhìn
    # ax.plot(freqs_hz, H_ecg_applied * max_ecg, color='royalblue', linestyle='--', label=f'High-Pass Filter H')
    # ax.axvline(x=ecg_fc_hz, color='orange', linestyle=':', linewidth=2, label=f'Cutoff ≈ {ecg_fc_hz:.2f} Hz')
    # ax.set_title("ECG (High-Pass): BEFORE Filtering", fontweight='bold')
    # ax.set_xlim(0, 15) # Zoom in on the 0-15Hz range
    # ax.set_ylim(ylim_ecg) # FIX Y-AXIS
    # ax.set_ylabel("Amplitude")
    # ax.legend(loc='upper right')
    # ax.grid(True, alpha=0.3)
    
    # # [Bottom-Left] ECG: AFTER (High-Pass Histogram)
    # ax = axes_spec[1, 0]
    # ax.bar(freqs_hz, filtered_ecg_fft, width=bar_width, color='royalblue', alpha=0.8, label='Filtered ECG Average')
    # ax.axvline(x=ecg_fc_hz, color='orange', linestyle=':', linewidth=2)
    # ax.set_title("ECG (High-Pass): AFTER Filtering", fontweight='bold')
    # ax.set_xlabel("Frequency (Hz)")
    # ax.set_ylabel("Amplitude")
    # ax.set_xlim(0, 15)
    # ax.set_ylim(ylim_ecg) # FIX Y-AXIS SAME AS ABOVE
    # ax.legend(loc='upper right')
    # ax.grid(True, alpha=0.3)
    
    # # [Top-Right] PPG: BEFORE (High-Pass Histogram)
    # ax = axes_spec[0, 1]
    # ax.bar(freqs_hz, avg_ppg_fft, width=bar_width, color='gray', alpha=0.7, label='Original PPG Average')
    # ax.plot(freqs_hz, H_ppg_applied * max_ppg, color='tomato', linestyle='--', label=f'High-Pass Filter H')
    # ax.axvline(x=ppg_fc_hz, color='orange', linestyle=':', linewidth=2, label=f'Cutoff ≈ {ppg_fc_hz:.2f} Hz')
    # ax.set_title("PPG (High-Pass): BEFORE Filtering", fontweight='bold')
    # ax.set_xlim(0, 15)
    # ax.set_ylim(ylim_ppg) # FIX Y-AXIS
    # ax.legend(loc='upper right')
    # ax.grid(True, alpha=0.3)
    
    # # [Bottom-Right] PPG: AFTER (High-Pass Histogram)
    # ax = axes_spec[1, 1]
    # ax.bar(freqs_hz, filtered_ppg_fft, width=bar_width, color='tomato', alpha=0.8, label='Filtered PPG Average')
    # ax.axvline(x=ppg_fc_hz, color='orange', linestyle=':', linewidth=2)
    # ax.set_title("PPG (High-Pass): AFTER Filtering", fontweight='bold')
    # ax.set_xlabel("Frequency (Hz)")
    # ax.set_xlim(0, 15)
    # ax.set_ylim(ylim_ppg) # FIX Y-AXIS SAME AS ABOVE
    # ax.legend(loc='upper right')
    # ax.grid(True, alpha=0.3)
    
    # fig_spectrum.suptitle("Average Frequency Spectrum (Histogram) Before and After Learned Filter", fontsize=18, fontweight='bold', color='navy')
    # fig_spectrum.tight_layout(rect=[0, 0, 1, 0.95])

    # ---------------------------------------------------------
    # 3. FUSED 3D (AFTER CROSS-ATTENTION)
    # ---------------------------------------------------------
    print(f"\n [3/4] Running t-SNE after CrossAttentionFusion (N={actual_samples})...")
    ppg_fused_all = np.concatenate(fused_3d_ppg_list, axis=0)[:actual_samples]
    ecg_fused_all = np.concatenate(fused_3d_ecg_list, axis=0)[:actual_samples]
    
    all_fused = np.vstack((ppg_fused_all, ecg_fused_all))
    
    tsne_fused = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=SEED)
    all_fused_2d = tsne_fused.fit_transform(all_fused)
    
    ppg_fused_2d = all_fused_2d[:actual_samples]
    ecg_fused_2d = all_fused_2d[actual_samples:]

    fig_fused = plt.figure(figsize=(10, 8))
    plt.scatter(ppg_fused_2d[:, 0], ppg_fused_2d[:, 1], color='tomato', label='PPG Fused 3D', alpha=0.7, s=50, edgecolors='w')
    plt.scatter(ecg_fused_2d[:, 0], ecg_fused_2d[:, 1], color='royalblue', label='ECG Fused 3D', alpha=0.7, s=50, edgecolors='w')

    for i in range(actual_samples):  
        plt.plot([ppg_fused_2d[i, 0], ecg_fused_2d[i, 0]], [ppg_fused_2d[i, 1], ecg_fused_2d[i, 1]], 
                 color='gray', alpha=0.2, linewidth=1, zorder=0)

    plt.title(f"Fused 3D Features (Cross-Attention) Alignment (N={actual_samples})", 
              fontsize=14, fontweight='bold', color='darkorange', pad=15)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.4)
    fig_fused.tight_layout()

    # ---------------------------------------------------------
    # 4. FINAL 128D EMBEDDINGS
    # ---------------------------------------------------------
    print(f"\n [4/4] Running t-SNE for final 128D embeddings (N={actual_samples})...")
    ppg_final = np.concatenate(final_ppg_embeddings, axis=0)[:actual_samples]
    ecg_final = np.concatenate(final_ecg_embeddings, axis=0)[:actual_samples]
    
    all_final = np.vstack((ppg_final, ecg_final))
    
    tsne_final = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=SEED)
    all_final_2d = tsne_final.fit_transform(all_final)
    
    ppg_2d_final = all_final_2d[:actual_samples]
    ecg_2d_final = all_final_2d[actual_samples:]

    fig2 = plt.figure(figsize=(10, 8))
    plt.scatter(ppg_2d_final[:, 0], ppg_2d_final[:, 1], color='tomato', label='PPG Final', alpha=0.7, s=50, edgecolors='w')
    plt.scatter(ecg_2d_final[:, 0], ecg_2d_final[:, 1], color='royalblue', label='ECG Final', alpha=0.7, s=50, edgecolors='w')

    for i in range(actual_samples):  
        plt.plot([ppg_2d_final[i, 0], ecg_2d_final[i, 0]], [ppg_2d_final[i, 1], ecg_2d_final[i, 1]], 
                 color='gray', alpha=0.2, linewidth=1, zorder=0)

    plt.title(f"Final 128D Space Alignment (N={actual_samples})", 
              fontsize=14, fontweight='bold', color='darkgreen', pad=15)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.4)
    fig2.tight_layout()

    print("Visualization ready! Displaying plots...")
    plt.show()

if __name__ == "__main__":
    visualize_feature_space()