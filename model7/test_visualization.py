import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

def visualize_latent_tsne(model, dataloader, device, num_batches=10, save_path="tsne_latent.png"):
    """
    Trích xuất latent features từ mô hình và vẽ biểu đồ t-SNE.
    
    Args:
        model: Mô hình Stage2_PPG2ECG_Improved đã được load trọng số.
        dataloader: Test hoặc Train dataloader.
        device: 'cuda' hoặc 'cpu'.
        num_batches: Số lượng batch để vẽ (t-SNE chạy khá chậm, không nên dùng toàn bộ dataset).
    """
    print("\n" + "="*50)
    print(" BẮT ĐẦU TRÍCH XUẤT VÀ VẼ t-SNE LATENT SPACE")
    print("="*50)
    
    model.eval()
    
    all_ppg_latents = []
    all_ecg_latents = []
    
    with torch.no_grad():
        loop = tqdm(enumerate(dataloader), total=min(len(dataloader), num_batches), desc="Extracting Features")
        for batch_idx, (ecg, ppg, _) in loop:
            if batch_idx >= num_batches:
                break
                
            ppg = ppg.float().to(device)
            ecg = ecg.float().to(device)
            if ppg.dim() == 2: ppg = ppg.unsqueeze(1)
            if ecg.dim() == 2: ecg = ecg.unsqueeze(1)
            
            # Truyền cả ecg_real để model trả về latent_ecg_teacher
            out = model(ppg, ecg_real=ecg)
            
            # Lấy latent tensors: Shape (Batch, Channels, Length)
            latent_ppg = out["latent_ppg"]
            latent_ecg = out["latent_ecg_teacher"]
            
            # Global Average Pooling để làm phẳng thành (Batch, Channels)
            latent_ppg_flat = F.adaptive_avg_pool1d(latent_ppg, 1).squeeze(-1)
            latent_ecg_flat = F.adaptive_avg_pool1d(latent_ecg, 1).squeeze(-1)
            
            # Chuyển sang Numpy
            all_ppg_latents.append(latent_ppg_flat.cpu().numpy())
            all_ecg_latents.append(latent_ecg_flat.cpu().numpy())

    # Gộp tất cả các batch lại
    ppg_features = np.concatenate(all_ppg_latents, axis=0)
    ecg_features = np.concatenate(all_ecg_latents, axis=0)
    num_samples = ppg_features.shape[0]
    
    print(f"\nĐã trích xuất {num_samples} mẫu. Đang chạy t-SNE (có thể mất vài phút)...")
    
    # Nối 2 tập feature lại để t-SNE fit trên cùng một không gian
    combined_features = np.vstack((ppg_features, ecg_features))
    
    # Cấu hình t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    tsne_results = tsne.fit_transform(combined_features)
    
    # Tách lại kết quả
    tsne_ppg = tsne_results[:num_samples, :]
    tsne_ecg = tsne_results[num_samples:, :]
    
    # --- VẼ BIỂU ĐỒ ---
    plt.figure(figsize=(10, 8))
    
    plt.scatter(tsne_ppg[:, 0], tsne_ppg[:, 1], c='green', label='PPG Latent', alpha=0.6, edgecolors='w', s=50)
    plt.scatter(tsne_ecg[:, 0], tsne_ecg[:, 1], c='blue', label='ECG Latent (Teacher)', alpha=0.6, edgecolors='w', s=50)
    
    plt.title('t-SNE Visualization of PPG and ECG Latent Spaces', fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ Đã lưu biểu đồ t-SNE tại: {save_path}")
    plt.show()

# ==========================================
# CÁCH CHẠY KHI TÍCH HỢP VÀO FILE CHÍNH
# ==========================================
if __name__ == "__main__":
    from load_data import LoadData
    from torch.utils.data import DataLoader
    from model import Stage1_ECG_AutoEncoder, Stage2_PPG2ECG_Improved
    import os
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    
    # 1. Load Dataloader
    test_loader = DataLoader(LoadData('/home/linhhima/Diffusion datasets/train.npz'), 
                             batch_size=BATCH_SIZE, shuffle=False)
                             
    # 2. Khởi tạo và Load Model
    pretrained_stage1 = Stage1_ECG_AutoEncoder(dims=[64, 128, 256]).to(device)
    # (Tuỳ chọn: load trọng số stage 1 nếu bạn cần tái tạo chính xác)
    stage1_path = "/home/linhhima/PPG_ECG/saved_models/best_stage1_ecg_ae.pth"
    if os.path.exists(stage1_path):
        pretrained_stage1.load_state_dict(torch.load(stage1_path, map_location=device))
        
    model = Stage2_PPG2ECG_Improved(pretrained_ecg_model=pretrained_stage1, dims=[64, 128, 256]).to(device)
    model_path = "/home/linhhima/PPG_ECG/saved_models/best_stage2_ppg2ecg.pth"
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # 3. Chạy hàm t-SNE
        # Lưu ý: Chỉnh num_batches tùy thuộc vào RAM máy tính, 10-20 batch (320-640 samples) là đủ để quan sát
        visualize_latent_tsne(model, test_loader, device, num_batches=15, save_path="tsne_latent_comparison.png")
    else:
        print(f"⚠️ Không tìm thấy trọng số model tại {model_path}!")