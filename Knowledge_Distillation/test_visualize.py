import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import os
from load_data import LoadData 
from CLIP import PPG2ECGModel

# ==========================================
SEED = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
INPUT_LENGTH = 2400
OUTPUT_EMBED_DIM = 128
TEST_DATA_PATH = '/home/linhhima/PPG_ECG/datasets/z_score_norm/total_mimic_af.npz'
CLIP_MODEL_PATH = "/home/linhhima/PPG_ECG/best_multitask_phase2.pth" 
SAMPLING_RATE = 125
NUM_SAMPLES_TO_VISUALIZE = 900

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_tsne_visualization():
    set_seed(SEED)
    
    print(f"1. Loading Model from {CLIP_MODEL_PATH}...")
    model = PPG2ECGModel(proj_dim=OUTPUT_EMBED_DIM).to(DEVICE)
    
    # Load trọng số an toàn
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

    all_z_ppg = []
    all_z_ecg = []
    samples_collected = 0

    print(f"3. Extracting Features for {NUM_SAMPLES_TO_VISUALIZE} samples...")
    with torch.no_grad():
        # unpack: ecg, ppg, record_name (hoặc nhãn)
        for ecg, ppg, *_ in tqdm(dataloader, desc="Forward Pass"):
            ecg_target = ecg.float().unsqueeze(1).to(DEVICE)
            ppg_input = ppg.float().unsqueeze(1).to(DEVICE)

            # Lấy dictionary kết quả
            outputs = model(ppg=ppg_input, ecg=ecg_target)
            
            all_z_ppg.append(outputs["z_ppg"].cpu().numpy())
            all_z_ecg.append(outputs["z_ecg"].cpu().numpy())

            samples_collected += ppg_input.shape[0]
            if samples_collected >= NUM_SAMPLES_TO_VISUALIZE:
                break
    
    # Gộp tất cả list thành numpy array và cắt đúng số lượng cần thiết
    z_ppg_np = np.concatenate(all_z_ppg, axis=0)[:NUM_SAMPLES_TO_VISUALIZE]
    z_ecg_np = np.concatenate(all_z_ecg, axis=0)[:NUM_SAMPLES_TO_VISUALIZE]

    # [QUAN TRỌNG] Chồng 2 tập dữ liệu lại với nhau trước khi chạy t-SNE 
    # để đảm bảo chúng được giáng chiều (project) trong CÙNG MỘT KHÔNG GIAN 2D
    features_combined = np.vstack((z_ppg_np, z_ecg_np))
    
    print("4. Running t-SNE dimension reduction (this may take a minute)...")
    # Perplexity thường để 30-50, random_state để giữ kết quả cố định qua các lần chạy
    tsne = TSNE(n_components=2, perplexity=40, random_state=SEED, init='pca', learning_rate='auto')
    embeddings_2d = tsne.fit_transform(features_combined)

    # Tách ngược lại thành toạ độ của PPG và ECG
    ppg_2d = embeddings_2d[:NUM_SAMPLES_TO_VISUALIZE, :]
    ecg_2d = embeddings_2d[NUM_SAMPLES_TO_VISUALIZE:, :]

    print("5. Plotting results...")
    plt.figure(figsize=(12, 10))
    
    # Vẽ điểm cho ECG (Ground Truth Representation)
    plt.scatter(ecg_2d[:, 0], ecg_2d[:, 1], c='blue', alpha=0.5, label='ECG Features (Target)', s=20, edgecolors='none')
    
    # Vẽ điểm cho PPG (Predicted Representation)
    plt.scatter(ppg_2d[:, 0], ppg_2d[:, 1], c='red', alpha=0.5, label='PPG Features (Fused)', s=20, edgecolors='none')

    # (Tùy chọn) Vẽ các đường nối mờ nhạt giữa PPG và ECG của cùng 1 cặp để xem chúng gần nhau không
    # Chỉ vẽ 100 đường nối để biểu đồ không bị quá rác
    for i in range(min(100, NUM_SAMPLES_TO_VISUALIZE)):
        plt.plot([ppg_2d[i, 0], ecg_2d[i, 0]], [ppg_2d[i, 1], ecg_2d[i, 1]], color='gray', alpha=0.15, linewidth=0.8)

    plt.title('t-SNE Visualization of PPG and ECG Latent Space', fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(fontsize=12, markerscale=2)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = "tsne_contrastive_alignment.png"
    plt.savefig(output_path, dpi=300)
    print(f"Done! Visualization saved to {output_path}")
    # plt.show() # Uncomment nếu bạn chạy trên Jupyter/Môi trường có UI

if __name__ == "__main__":
    run_tsne_visualization()