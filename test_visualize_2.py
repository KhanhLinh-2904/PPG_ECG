import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE  # Thêm thư viện TSNE
import torch.nn.functional as F    # Thêm F cho normalize
from load_data import LoadData
from CLIP_2 import ECGEssembleCLIP  
import random
import os
from metric import calculate_cosine_similarity, calculate_dtw_distance, calculate_metrics

# --- CONFIGURATION ---
SEED = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
INPUT_LENGTH = 2400
OUTPUT_EMBED_DIM = 128
TEST_DATA_PATH = 'processed_data/mimic3_v1_2400_test.npz'
CLIP_MODEL_PATH = "multitask_best_model.pth" 

# Cấu hình cho t-SNE
NUM_SAMPLES_TO_VISUALIZE = 1000 
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
    """
    Hàm mới: Trích xuất và vẽ t-SNE cho các Layer trung gian và Final Embedding 128D
    Hiển thị ra 2 cửa sổ đồ thị.
    """
    set_seed(SEED)
    model = load_model()
    
    try:
        test_dataset = LoadData(TEST_DATA_PATH)
        # Bắt buộc shuffle=False để các điểm PPG và ECG khớp theo cặp Index
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    except Exception as e:
        print(f"Error loading data for t-SNE: {e}")
        return

    # --- 1. CÀI ĐẶT HOOKS ---
    activations = {layer: {'ppg': [], 'ecg': []} for layer in LAYERS_TO_VISUALIZE}

    def get_activation(layer_name, modality):
        def hook(module, input, output):
            out_tensor = output[0] if isinstance(output, tuple) else output
            if len(out_tensor.shape) == 3:
                out_tensor = out_tensor.mean(dim=2) # GAP 1D
            activations[layer_name][modality].append(out_tensor.detach().cpu().numpy())
        return hook

    try:
        # Giả định cấu trúc DualDomainEncoder có thuộc tính time_branch
        for layer_name in LAYERS_TO_VISUALIZE:
            getattr(model.encode_ppg.time_branch, layer_name).register_forward_hook(get_activation(layer_name, 'ppg'))
            getattr(model.encode_ecg.time_branch, layer_name).register_forward_hook(get_activation(layer_name, 'ecg'))
        print("✅ Hooks registered successfully on ResNet1D layers.")
    except AttributeError as e:
        print(f"❌ Error setting hooks: {e}")
        print("Vui lòng kiểm tra lại đường dẫn tới ResNet layers trong ECGEssembleCLIP.")
        return

    # --- 2. TRÍCH XUẤT ĐẶC TRƯNG ---
    print(f"⏳ Extracting features for max {NUM_SAMPLES_TO_VISUALIZE} samples...")
    collected_samples = 0
    final_ppg_embeddings = []
    final_ecg_embeddings = []

    with torch.no_grad():
        for ecg, ppg, _ in test_loader:
            if collected_samples >= NUM_SAMPLES_TO_VISUALIZE:
                break
                
            ecg_input = ecg.to(DEVICE).float().unsqueeze(1)
            ppg_input = ppg.to(DEVICE).float().unsqueeze(1)
            
            # Forward toàn bộ để lấy Final Embedding và kích hoạt Hooks
            ecg_embed, ppg_embed, _ = model(ecg_input, ppg_input)
            
            # L2 Normalize nếu hàm forward chưa chuẩn hoá
            ecg_embed = F.normalize(ecg_embed, p=2, dim=-1)
            ppg_embed = F.normalize(ppg_embed, p=2, dim=-1)

            final_ecg_embeddings.append(ecg_embed.cpu().numpy())
            final_ppg_embeddings.append(ppg_embed.cpu().numpy())
            
            collected_samples += ppg_input.size(0)

    actual_samples = np.concatenate(final_ppg_embeddings, axis=0)[:NUM_SAMPLES_TO_VISUALIZE].shape[0]

    # --- 3. VẼ ĐỒ THỊ 1: TỪNG LAYER ---
    print("\n🧠 [1/2] Running t-SNE for 4 intermediate layers...")
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

    # --- 4. VẼ ĐỒ THỊ 2: FINAL EMBEDDING ---
    print(f"\n🧠 [2/2] Running t-SNE for final 128D embeddings (N={actual_samples})...")
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

    print("✅ Visualization ready! Displaying plots...")
    plt.show()

if __name__ == "__main__":
   
    visualize_feature_space()
   