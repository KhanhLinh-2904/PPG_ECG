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
TEST_DATA_PATH = 'processed_data/mimic3_v1_2400_test.npz'
CLIP_MODEL_PATH = "multitask_best_model.pth" 
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

    freq_ppg_list = []
    freq_ecg_list = []

    sum_ecg_fft = None
    sum_ppg_fft = None

    raw_fft_ppg_list = []
    raw_fft_ecg_list = []
    with torch.no_grad():
        for ecg, ppg, _ in test_loader:
            if collected_samples >= NUM_SAMPLES_TO_VISUALIZE:
                break
                
            ecg_input = ecg.to(DEVICE).float().unsqueeze(1)
            ppg_input = ppg.to(DEVICE).float().unsqueeze(1)
            
            raw_fft_ecg_list.append(torch.abs(torch.fft.rfft(ecg_input.squeeze(1))).cpu().numpy())
            raw_fft_ppg_list.append(torch.abs(torch.fft.rfft(ppg_input.squeeze(1))).cpu().numpy())
            batch_ecg_fft = torch.abs(torch.fft.rfft(ecg_input.squeeze(1))).sum(dim=0).cpu().numpy()
            batch_ppg_fft = torch.abs(torch.fft.rfft(ppg_input.squeeze(1))).sum(dim=0).cpu().numpy()
            
            if sum_ecg_fft is None:
                sum_ecg_fft = batch_ecg_fft
                sum_ppg_fft = batch_ppg_fft
            else:
                sum_ecg_fft += batch_ecg_fft
                sum_ppg_fft += batch_ppg_fft

            # Trích xuất fused_3d
            ecg_fused_1d, ecg_fused_3d, ecg_features_list = model.encode_ecg(ecg_input)
            ppg_fused_1d, ppg_fused_3d, ppg_features_list = model.encode_ppg(ppg_input)

            ecg_fre_feat = model.encode_ecg.freq_branch(ecg_input)
            ppg_fre_feat = model.encode_ppg.freq_branch(ppg_input)

            # Lấy trung bình theo chiều dài (Global Average Pooling) để đưa về 2D cho t-SNE
            ecg_f3d_pooled = ecg_fused_3d.mean(dim=2).cpu().numpy()
            ppg_f3d_pooled = ppg_fused_3d.mean(dim=2).cpu().numpy()
            
            fused_3d_ecg_list.append(ecg_f3d_pooled)
            fused_3d_ppg_list.append(ppg_f3d_pooled)

            freq_ecg_list.append(ecg_fre_feat.cpu().numpy())
            freq_ppg_list.append(ppg_fre_feat.cpu().numpy())

            # Trích xuất final embedding
            ecg_embed, ppg_embed, _ = model(ecg_input, ppg_input)
            
            ecg_embed = F.normalize(ecg_embed, p=2, dim=-1)
            ppg_embed = F.normalize(ppg_embed, p=2, dim=-1)

            final_ecg_embeddings.append(ecg_embed.cpu().numpy())
            final_ppg_embeddings.append(ppg_embed.cpu().numpy())
            
            collected_samples += ppg_input.size(0)

    actual_samples = np.concatenate(final_ppg_embeddings, axis=0)[:NUM_SAMPLES_TO_VISUALIZE].shape[0]

    # ---------------------------------------------------------
    # 1. VẼ ĐỒ THỊ LAYERS 1 -> 4
    # ---------------------------------------------------------
    print("\n [1/5] Running t-SNE for 4 intermediate layers...")
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
    # ---------------------------------------------------------
    # 2. FREQUENCY FEATURES
    # ---------------------------------------------------------
    print(f"\n [2/5] Running t-SNE for Frequency Features (N={actual_samples})...")
    ppg_freq_all = np.concatenate(freq_ppg_list, axis=0)[:actual_samples]
    ecg_freq_all = np.concatenate(freq_ecg_list, axis=0)[:actual_samples]
    
    all_freq = np.vstack((ppg_freq_all, ecg_freq_all))
    
    tsne_freq = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=SEED)
    all_freq_2d = tsne_freq.fit_transform(all_freq)
    
    ppg_freq_2d = all_freq_2d[:actual_samples]
    ecg_freq_2d = all_freq_2d[actual_samples:]

    fig_freq = plt.figure(figsize=(10, 8))
    plt.scatter(ppg_freq_2d[:, 0], ppg_freq_2d[:, 1], color='tomato', label='PPG Freq Feature', alpha=0.7, s=50, edgecolors='w')
    plt.scatter(ecg_freq_2d[:, 0], ecg_freq_2d[:, 1], color='royalblue', label='ECG Freq Feature', alpha=0.7, s=50, edgecolors='w')

    for i in range(actual_samples):  
        plt.plot([ppg_freq_2d[i, 0], ecg_freq_2d[i, 0]], [ppg_freq_2d[i, 1], ecg_freq_2d[i, 1]], 
                 color='gray', alpha=0.2, linewidth=1, zorder=0)

    plt.title(f"Frequency Features Alignment (N={actual_samples})\n(Expected: Separation due to mechanical vs electrical nature)", 
              fontsize=14, fontweight='bold', color='purple', pad=15)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.4)
    fig_freq.tight_layout()
    # ---------------------------------------------------------
    # 3.  FUSED 3D (AFTER CROSS-ATTENTION)
    # ---------------------------------------------------------
    print(f"\n [3/5] Running t-SNE for Fused 3D Features (N={actual_samples})...")
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
    print(f"\n [4/5] Running t-SNE for final 128D embeddings (N={actual_samples})...")
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

    plt.title(f"Final 128D Space Alignment (N={actual_samples})\n(Ideal: mixed dots, short lines)", 
              fontsize=14, fontweight='bold', color='darkgreen', pad=15)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.4)
    fig2.tight_layout()

    # ---------------------------------------------------------
    # 5. RAW AVERAGE FFT SPECTRUM
    # ---------------------------------------------------------

    print(f"\n [6/6] Running t-SNE for Raw FFT Features (N={actual_samples})...")
    
    # Gộp list thành mảng numpy lớn
    ppg_raw_fft_all = np.concatenate(raw_fft_ppg_list, axis=0)[:actual_samples]
    ecg_raw_fft_all = np.concatenate(raw_fft_ecg_list, axis=0)[:actual_samples]
    
    # Loại bỏ thành phần DC (cột 0) để không làm nhiễu t-SNE
    ppg_raw_fft_all = ppg_raw_fft_all[:, 1:]
    ecg_raw_fft_all = ecg_raw_fft_all[:, 1:]
    
    all_raw_fft = np.vstack((ppg_raw_fft_all, ecg_raw_fft_all))
    
    # Chạy t-SNE
    tsne_raw_fft = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=SEED)
    all_raw_fft_2d = tsne_raw_fft.fit_transform(all_raw_fft)
    
    ppg_raw_fft_2d = all_raw_fft_2d[:actual_samples]
    ecg_raw_fft_2d = all_raw_fft_2d[actual_samples:]

    fig_raw_fft_tsne = plt.figure(figsize=(10, 8))
    plt.scatter(ppg_raw_fft_2d[:, 0], ppg_raw_fft_2d[:, 1], color='tomato', label='PPG Raw FFT', alpha=0.7, s=50, edgecolors='w')
    plt.scatter(ecg_raw_fft_2d[:, 0], ecg_raw_fft_2d[:, 1], color='royalblue', label='ECG Raw FFT', alpha=0.7, s=50, edgecolors='w')

    for i in range(actual_samples):  
        plt.plot([ppg_raw_fft_2d[i, 0], ecg_raw_fft_2d[i, 0]], [ppg_raw_fft_2d[i, 1], ecg_raw_fft_2d[i, 1]], 
                 color='gray', alpha=0.2, linewidth=1, zorder=0)

    plt.title(f"Raw FFT Space Alignment (N={actual_samples})\n(Expected: Complete Separation)", 
              fontsize=14, fontweight='bold', color='maroon', pad=15)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.4)
    fig_raw_fft_tsne.tight_layout()


    print("Visualization ready! Displaying plots...")
    plt.show()

if __name__ == "__main__":
    visualize_feature_space()