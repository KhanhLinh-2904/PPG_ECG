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
NUM_SAMPLES_TO_VISUALIZE = 1000 # Số lượng mẫu để t-SNE chạy mượt mà
INPUT_LENGTH = 2400
OUTPUT_EMBED_DIM = 128 # Kích thước vector cuối cùng
TARGET_LENGTH = 2400
TEST_DATA_PATH = 'processed_data/mimic3_v1_2400_train.npz'
CLIP_MODEL_PATH = "multitask_clip_best_model.pth"

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
        print("✅ CLIP weights loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
    
    model_clip.eval()
    return model_clip

def visualize_final_embeddings():
    model_clip = load_models()
    
    if not os.path.exists(TEST_DATA_PATH):
        print(f"❌ File {TEST_DATA_PATH} không tồn tại!")
        return

    test_dataset = LoadData(TEST_DATA_PATH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    ppg_embeddings = []
    ecg_embeddings = []
    
    print(f"⏳ Extracting final {OUTPUT_EMBED_DIM}D features for {NUM_SAMPLES_TO_VISUALIZE} samples...")
    collected_samples = 0
    
    with torch.no_grad():
        for ppg_real, ecg_real, _ in test_loader:
            if collected_samples >= NUM_SAMPLES_TO_VISUALIZE:
                break
                
            ppg_real = ppg_real.to(DEVICE).float().unsqueeze(1)
            ecg_real = ecg_real.to(DEVICE).float().unsqueeze(1)
            
            # Lấy vector embedding 128 chiều cuối cùng (sau lớp FC)
            ppg_embed = model_clip.encode_ppg(ppg_real)
            ecg_embed = model_clip.encode_ecg(ecg_real)
            
            # Xử lý nếu trả về tuple
            ppg_embed_tensor = ppg_embed[0] if isinstance(ppg_embed, tuple) else ppg_embed
            ecg_embed_tensor = ecg_embed[0] if isinstance(ecg_embed, tuple) else ecg_embed

            ppg_embed_tensor = F.normalize(ppg_embed_tensor, p=2, dim=-1)
            ecg_embed_tensor = F.normalize(ecg_embed_tensor, p=2, dim=-1)

            ppg_embeddings.append(ppg_embed_tensor.cpu().numpy())
            ecg_embeddings.append(ecg_embed_tensor.cpu().numpy())
            
            collected_samples += ppg_real.size(0)

    ppg_all = np.concatenate(ppg_embeddings, axis=0)[:NUM_SAMPLES_TO_VISUALIZE]
    ecg_all = np.concatenate(ecg_embeddings, axis=0)[:NUM_SAMPLES_TO_VISUALIZE]
    
    # SỬA LỖI Ở ĐÂY: Lấy số lượng mẫu thực tế thu thập được
    actual_samples = ppg_all.shape[0] 
    
    # Gom chung để t-SNE chiếu trên cùng 1 mặt phẳng
    all_features = np.vstack((ppg_all, ecg_all))
    
    print(f"🧠 Running t-SNE on {actual_samples} samples... (This might take a few seconds)")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=SEED)
    all_features_2d = tsne.fit_transform(all_features)
    
    # Tách lại ra PPG và ECG dựa trên số lượng thực tế
    ppg_2d = all_features_2d[:actual_samples]
    ecg_2d = all_features_2d[actual_samples:]

    # --- VISUALIZATION ---
    plt.figure(figsize=(10, 8))
    
    # Vẽ các điểm
    plt.scatter(ppg_2d[:, 0], ppg_2d[:, 1], color='tomato', label='PPG Final Embedding', alpha=0.7, s=50, edgecolors='w')
    plt.scatter(ecg_2d[:, 0], ecg_2d[:, 1], color='royalblue', label='ECG Final Embedding', alpha=0.7, s=50, edgecolors='w')

    # Vẽ đường nối các cặp của cùng 1 người
    for i in range(actual_samples):  # Dùng actual_samples thay vì biến cấu hình
        plt.plot([ppg_2d[i, 0], ecg_2d[i, 0]], [ppg_2d[i, 1], ecg_2d[i, 1]], 
                 color='gray', alpha=0.2, linewidth=1, zorder=0)

    plt.title(f"Final 128D Embedding Space Alignment (N={actual_samples})\n(If dots mix well and lines are short, the model succeeded!)", 
              fontsize=14, fontweight='bold', color='darkgreen', pad=15)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_final_embeddings()