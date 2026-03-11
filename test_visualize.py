import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from load_data import LoadData
from CLIP import ECGDecoder_UNet, ECGEssembleCLIP
import torch.nn.functional as F 

# --- CONFIGURATION ---
SEED = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NUM_SAMPLES_TO_VISUALIZE = 300 # Số lượng mẫu (có thể tăng lên 1000 nếu máy bạn mạnh)
INPUT_LENGTH = 2400
OUTPUT_EMBED_DIM = 128
TARGET_LENGTH = 2400
TEST_DATA_PATH = 'processed_data/mimic3_v1_2400_train.npz'
CLIP_MODEL_PATH = "multitask_clip_best_model.pth"
# DECODER_MODEL_PATH = "multitask_decoder_best_model.pth"

# Danh sách các layer muốn lấy feature của ResNet50_1D
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

def visualize_tsne_layers():
    model_clip = load_models()
    
    if not os.path.exists(TEST_DATA_PATH):
        print(f"❌ File {TEST_DATA_PATH} không tồn tại!")
        return

    test_dataset = LoadData(TEST_DATA_PATH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 1. CÀI ĐẶT HOOKS ĐỂ LẤY FEATURE TRUNG GIAN ---
    activations = {layer: {'ppg': [], 'ecg': []} for layer in LAYERS_TO_VISUALIZE}

    def get_activation(layer_name, modality):
        def hook(model, input, output):
            # output của các layer1..4 trong ResNet1D là tensor dạng [Batch, Channels, Length]
            out_tensor = output[0] if isinstance(output, tuple) else output
            
            # Global Average Pooling 1D: Lấy trung bình theo chiều dài (Length)
            # Biến đổi [Batch, Channels, Length] -> [Batch, Channels]
            if len(out_tensor.shape) == 3:
                out_tensor = out_tensor.mean(dim=2) 
                
            activations[layer_name][modality].append(out_tensor.detach().cpu().numpy())
        return hook

    # ĐĂNG KÝ HOOK VÀO ĐÚNG TÊN BIẾN
    # Dựa vào cấu trúc, mình gọi ppg_encoder và ecg_encoder. 
    # Nếu trong ECGEssembleCLIP bạn đặt tên khác (ví dụ: model_ppg, model_ecg), hãy sửa 2 biến dưới đây!
    
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
        print(f"❌ Error: {e}")
        print("Hãy kiểm tra lại tên biến 'ppg_encoder' và 'ecg_encoder' trong file CLIP.py của bạn.")
        return

    print(f"⏳ Extracting intermediate features for {NUM_SAMPLES_TO_VISUALIZE} samples...")
    collected_samples = 0
    
    # --- 2. CHẠY INFERENCE ---
    with torch.no_grad():
        for ppg_real, ecg_real, _ in test_loader:
            if collected_samples >= NUM_SAMPLES_TO_VISUALIZE:
                break
                
            ppg_real = ppg_real.to(DEVICE).float().unsqueeze(1)
            ecg_real = ecg_real.to(DEVICE).float().unsqueeze(1)
            
            # Quá trình này sẽ kích hoạt các Hooks và tự động lưu data vào 'activations'
            _ = model_clip.encode_ppg(ppg_real)
            _ = model_clip.encode_ecg(ecg_real)
            
            collected_samples += ppg_real.size(0)

    # --- 3. CHẠY t-SNE VÀ VẼ ĐỒ THỊ ---
    fig, axes = plt.subplots(2, 2, figsize=(18, 14)) # 2 hàng x 2 cột cho 4 layers
    axes = axes.flatten()

    print("🧠 Running t-SNE for each layer (this will take a moment)...")
    
    for idx, layer_name in enumerate(LAYERS_TO_VISUALIZE):
        ax = axes[idx]
        print(f"   -> Processing {layer_name}...")
        
        # Gộp tất cả các batch lại (shape: [N, Channels])
        ppg_layer_feats = np.concatenate(activations[layer_name]['ppg'], axis=0)[:NUM_SAMPLES_TO_VISUALIZE]
        ecg_layer_feats = np.concatenate(activations[layer_name]['ecg'], axis=0)[:NUM_SAMPLES_TO_VISUALIZE]
        
        all_features = np.vstack((ppg_layer_feats, ecg_layer_feats))
        
        # Chạy t-SNE
        tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=SEED)
        all_features_2d = tsne.fit_transform(all_features)
        
        # Tách lại ra PPG và ECG
        ppg_2d = all_features_2d[:NUM_SAMPLES_TO_VISUALIZE]
        ecg_2d = all_features_2d[NUM_SAMPLES_TO_VISUALIZE:]

        # Vẽ Scatter Plot
        ax.scatter(ppg_2d[:, 0], ppg_2d[:, 1], color='tomato', label='PPG', alpha=0.7, s=40, edgecolors='w')
        ax.scatter(ecg_2d[:, 0], ecg_2d[:, 1], color='royalblue', label='ECG', alpha=0.7, s=40, edgecolors='w')

        # Vẽ đường nối giữa PPG và ECG của cùng 1 người
        for i in range(NUM_SAMPLES_TO_VISUALIZE):
            ax.plot([ppg_2d[i, 0], ecg_2d[i, 0]], [ppg_2d[i, 1], ecg_2d[i, 1]], 
                    color='gray', alpha=0.1, linewidth=0.5, zorder=0)

        # Lấy kích thước channel của layer hiện tại để hiển thị lên tiêu đề
        channel_dim = ppg_layer_feats.shape[1]
        ax.set_title(f"After {layer_name.upper()} (Dim: {channel_dim})", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)

    plt.suptitle("Evolution of PPG and ECG Embeddings through ResNet50 Layers", 
                 fontsize=20, fontweight='bold', color='darkred')
    plt.tight_layout(rect=[0, 0, 1, 0.95]) 
    plt.show()

if __name__ == "__main__":
    visualize_tsne_layers()