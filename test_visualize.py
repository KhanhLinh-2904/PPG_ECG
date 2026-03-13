import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from load_data import LoadData
from CLIP import ECGEssembleCLIP
import torch.nn.functional as F 

# --- CONFIGURATION ---
SEED = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NUM_SAMPLES_TO_VISUALIZE = 1000 # Số lượng mẫu để t-SNE chạy mượt mà
INPUT_LENGTH = 2400
OUTPUT_EMBED_DIM = 128
TARGET_LENGTH = 2400
TEST_DATA_PATH = 'processed_data/mimic3_v1_2400_train.npz'
CLIP_MODEL_PATH = "multitask_clip_best_model.pth"

# Danh sách các layer muốn lấy feature
LAYERS_TO_VISUALIZE = ['layer1', 'layer2', 'layer3', 'layer4']

# Thiết lập seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def load_models():
    print(f"🚀 Loading models on {DEVICE}...")
    model_clip = ECGEssembleCLIP(embed_dim=OUTPUT_EMBED_DIM).to(DEVICE)
    try:
        clip_weights = torch.load(CLIP_MODEL_PATH, map_location=DEVICE)
        model_clip.load_state_dict(clip_weights)
        print("✅ Weights loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
    model_clip.eval()
    return model_clip

def visualize_all_features():
    model_clip = load_models()
    
    if not os.path.exists(TEST_DATA_PATH):
        print(f"❌ File {TEST_DATA_PATH} không tồn tại!")
        return

    test_dataset = LoadData(TEST_DATA_PATH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ==========================================
    # 1. CÀI ĐẶT HOOKS ĐỂ LẤY FEATURE TRUNG GIAN
    # ==========================================
    activations = {layer: {'ppg': [], 'ecg': []} for layer in LAYERS_TO_VISUALIZE}

    def get_activation(layer_name, modality):
        def hook(model, input, output):
            out_tensor = output[0] if isinstance(output, tuple) else output
            # Global Average Pooling 1D
            if len(out_tensor.shape) == 3:
                out_tensor = out_tensor.mean(dim=2) 
            activations[layer_name][modality].append(out_tensor.detach().cpu().numpy())
        return hook

    PPG_ENCODER_NAME = 'encode_ppg' 
    ECG_ENCODER_NAME = 'encode_ecg'
    
    try:
        ppg_module = getattr(model_clip, PPG_ENCODER_NAME)
        ecg_module = getattr(model_clip, ECG_ENCODER_NAME)
        
        for layer_name in LAYERS_TO_VISUALIZE:
            getattr(ppg_module, layer_name).register_forward_hook(get_activation(layer_name, 'ppg'))
            getattr(ecg_module, layer_name).register_forward_hook(get_activation(layer_name, 'ecg'))
        print("✅ Hooks registered successfully on ResNet50_1D layers!")
    except AttributeError as e:
        print(f"❌ Error setting hooks: {e}")
        return

    # ==========================================
    # 2. CHẠY INFERENCE (1 LẦN DUY NHẤT)
    # ==========================================
    print(f"⏳ Extracting BOTH intermediate and final {OUTPUT_EMBED_DIM}D features for {NUM_SAMPLES_TO_VISUALIZE} samples...")
    collected_samples = 0
    
    final_ppg_embeddings = []
    final_ecg_embeddings = []
    
    with torch.no_grad():
        for ppg_real, ecg_real, _ in test_loader:
            if collected_samples >= NUM_SAMPLES_TO_VISUALIZE:
                break
                
            ppg_real = ppg_real.to(DEVICE).float().unsqueeze(1)
            ecg_real = ecg_real.to(DEVICE).float().unsqueeze(1)
            
            # Forward pass: Kích hoạt Hooks (Lưu Layer 1->4) VÀ lấy Final Embedding
            ppg_embed = model_clip.encode_ppg(ppg_real)
            ecg_embed = model_clip.encode_ecg(ecg_real)
            
            # Xử lý Final Embedding
            ppg_embed_tensor = ppg_embed[0] if isinstance(ppg_embed, tuple) else ppg_embed
            ecg_embed_tensor = ecg_embed[0] if isinstance(ecg_embed, tuple) else ecg_embed

            ppg_embed_tensor = F.normalize(ppg_embed_tensor, p=2, dim=-1)
            ecg_embed_tensor = F.normalize(ecg_embed_tensor, p=2, dim=-1)

            final_ppg_embeddings.append(ppg_embed_tensor.cpu().numpy())
            final_ecg_embeddings.append(ecg_embed_tensor.cpu().numpy())
            
            collected_samples += ppg_real.size(0)

    # Chốt số lượng mẫu thực tế
    actual_samples = np.concatenate(final_ppg_embeddings, axis=0)[:NUM_SAMPLES_TO_VISUALIZE].shape[0]

    # ==========================================
    # 3. VẼ ĐỒ THỊ 1: TỪNG LAYER (T-SNE)
    # ==========================================
    print("🧠 [1/2] Running t-SNE for intermediate layers...")
    fig1, axes = plt.subplots(2, 2, figsize=(18, 14)) 
    axes = axes.flatten()
    
    for idx, layer_name in enumerate(LAYERS_TO_VISUALIZE):
        ax = axes[idx]
        print(f"   -> Processing {layer_name}...")
        
        ppg_layer_feats = np.concatenate(activations[layer_name]['ppg'], axis=0)[:actual_samples]
        ecg_layer_feats = np.concatenate(activations[layer_name]['ecg'], axis=0)[:actual_samples]
        
        all_features = np.vstack((ppg_layer_feats, ecg_layer_feats))
        
        tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=SEED)
        all_features_2d = tsne.fit_transform(all_features)
        
        ppg_2d = all_features_2d[:actual_samples]
        ecg_2d = all_features_2d[actual_samples:]

        ax.scatter(ppg_2d[:, 0], ppg_2d[:, 1], color='tomato', label='PPG', alpha=0.7, s=40, edgecolors='w')
        ax.scatter(ecg_2d[:, 0], ecg_2d[:, 1], color='royalblue', label='ECG', alpha=0.7, s=40, edgecolors='w')

        for i in range(actual_samples):
            ax.plot([ppg_2d[i, 0], ecg_2d[i, 0]], [ppg_2d[i, 1], ecg_2d[i, 1]], 
                    color='gray', alpha=0.1, linewidth=0.5, zorder=0)

        channel_dim = ppg_layer_feats.shape[1]
        ax.set_title(f"After {layer_name.upper()} (Dim: {channel_dim})", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)

    fig1.suptitle("Evolution of PPG and ECG Embeddings through ResNet50 Layers", 
                 fontsize=20, fontweight='bold', color='darkred')
    fig1.tight_layout(rect=[0, 0, 1, 0.95]) 

    # ==========================================
    # 4. VẼ ĐỒ THỊ 2: FINAL EMBEDDING 128D
    # ==========================================
    print(f"🧠 [2/2] Running t-SNE for final 128D embeddings (N={actual_samples})...")
    ppg_all_final = np.concatenate(final_ppg_embeddings, axis=0)[:actual_samples]
    ecg_all_final = np.concatenate(final_ecg_embeddings, axis=0)[:actual_samples]
    
    all_final_features = np.vstack((ppg_all_final, ecg_all_final))
    
    tsne_final = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=SEED)
    all_final_2d = tsne_final.fit_transform(all_final_features)
    
    ppg_2d_final = all_final_2d[:actual_samples]
    ecg_2d_final = all_final_2d[actual_samples:]

    fig2 = plt.figure(figsize=(10, 8))
    plt.scatter(ppg_2d_final[:, 0], ppg_2d_final[:, 1], color='tomato', label='PPG Final Embedding', alpha=0.7, s=50, edgecolors='w')
    plt.scatter(ecg_2d_final[:, 0], ecg_2d_final[:, 1], color='royalblue', label='ECG Final Embedding', alpha=0.7, s=50, edgecolors='w')

    for i in range(actual_samples):  
        plt.plot([ppg_2d_final[i, 0], ecg_2d_final[i, 0]], [ppg_2d_final[i, 1], ecg_2d_final[i, 1]], 
                 color='gray', alpha=0.2, linewidth=1, zorder=0)

    plt.title(f"Final 128D Embedding Space Alignment (N={actual_samples})\n(If dots mix well and lines are short, the model succeeded!)", 
              fontsize=14, fontweight='bold', color='darkgreen', pad=15)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.4)
    fig2.tight_layout()

    # ==========================================
    # 5. HIỂN THỊ CẢ 2 ẢNH CÙNG LÚC
    # ==========================================
    print("✅ Done! Displaying plots...")
    plt.show()

if __name__ == "__main__":
    visualize_all_features()