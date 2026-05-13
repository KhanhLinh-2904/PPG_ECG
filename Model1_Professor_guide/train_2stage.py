import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from tqdm import tqdm
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random

# Import lớp DistributionAlignmentLoss thay cho ContrastiveLoss
from loss import DistributionAlignmentLoss, PearsonLoss 
from CLIP import ECG_PPG_Fusion_Model
from load_data import LoadData
import os

# ==========================================
# CẤU HÌNH CHẾ ĐỘ HUẤN LUYỆN (TRAINING MODE)
# ==========================================
TRAIN_PHASE_1 = False                     # True: Huấn luyện GĐ1 | False: Bỏ qua GĐ1
TRAIN_PHASE_2 = True                      # True: Huấn luyện GĐ2 | False: Bỏ qua GĐ2
PHASE1_CHECKPOINT = "best_phase1_reconstruction.pth" # Đường dẫn load model nếu bỏ qua GĐ1

# ==========================================
# CONSTANTS & HYPERPARAMETERS
# ==========================================
SEED = 44
EPOCHS_PHASE1 = 50       # Giai đoạn 1: Autoencoder (ECG)
EPOCHS_PHASE2 = 100      # Giai đoạn 2: Distribution Alignment (PPG -> ECG)
BATCH_SIZE = 64
LR_CNN = 1e-4           
LR_TRANSFORMER = 5e-4    

WEIGHT_MSE = 1.0      
WEIGHT_PEARSON = 1.0

# Trọng số cho Distribution Alignment Loss ở Phase 2
ALPHA_CORAL = 1.0        
BETA_KL = 1.0            

OUTPUT_EMBED_DIM = 128
SEQ_LENGTH = 2400

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# UTILITIES
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def plot_best_similarity_matrix(embed_ecg, embed_ppg, epoch):
    embed_ecg = F.normalize(embed_ecg, dim=1)
    embed_ppg = F.normalize(embed_ppg, dim=1)
    
    raw_sim = (embed_ppg @ embed_ecg.t()).detach().cpu().numpy()
    plt.figure(figsize=(8, 6))
    plt.imshow(raw_sim, cmap='viridis', vmin=-1, vmax=1)
    plt.colorbar(label='Cosine Similarity')
    plt.title(f"PPG-ECG Similarity Matrix (Phase 2 - Epoch: {epoch})")
    plt.xlabel("ECG Index")
    plt.ylabel("PPG Index")
    plt.savefig(f"similarity_matrix_epoch_{epoch}.png")
    plt.close()

def plot_losses(history, title, filename):
    plt.figure(figsize=(10, 6))
    for label, values in history.items():
        if values:
            plt.plot(range(1, len(values) + 1), values, label=label)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

# ==========================================
# PHASE 1: TRAIN ECG AUTOENCODER (RECONSTRUCTION)
# ==========================================
def train_phase1(model, dataloader, optimizer, mse_fn, pearson_fn, device, scaler, current_epoch):
    model.train()
    metrics = {'mse': 0.0, 'pearson': 0.0, 'total': 0.0}
    progress_bar = tqdm(dataloader, desc=f"Phase 1 - Epoch {current_epoch}", unit="batch")
    
    for ecg, _, _ in progress_bar:
        ecg_target = ecg.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        
        with autocast():
            # Chỉ đi qua ECG nhánh
            _, z_ecg_3d = model.encode_ecg(ecg_target)
            ecg_pred = model.decoder(z_ecg_3d)
            
            m_loss = mse_fn(ecg_pred, ecg_target)
            p_loss = pearson_fn(ecg_pred, ecg_target)
            total_loss = (WEIGHT_MSE * m_loss) + (WEIGHT_PEARSON * p_loss)

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(list(model.encode_ecg.parameters()) + list(model.decoder.parameters()), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        metrics['mse'] += m_loss.item()
        metrics['pearson'] += p_loss.item()
        metrics['total'] += total_loss.item()

        progress_bar.set_postfix({'MSE': f"{m_loss.item():.4f}", 'Pearson': f"{p_loss.item():.4f}"})

    return {k: v / len(dataloader) for k, v in metrics.items()}

def eval_phase1(model, dataloader, mse_fn, pearson_fn, device):
    model.eval()
    t_mse, t_pearson = 0.0, 0.0
    with torch.no_grad():
        for ecg, _, _ in dataloader:
            ecg_target = ecg.to(device).float().unsqueeze(1)
            _, z_ecg_3d = model.encode_ecg(ecg_target)
            ecg_pred = model.decoder(z_ecg_3d)
            
            t_mse += mse_fn(ecg_pred, ecg_target).item()
            t_pearson += pearson_fn(ecg_pred, ecg_target).item()
            
    n = len(dataloader)
    return t_mse/n, 1.0 - (t_pearson/n)

# ==========================================
# PHASE 2: TRAIN PPG ENCODER (DISTRIBUTION ALIGNMENT)
# ==========================================
def train_phase2(model, dataloader, optimizer, dist_align_fn, device, scaler, current_epoch):
    model.encode_ppg.train()
    model.encode_ecg.eval() # ECG đã đóng băng
    
    metrics = {'total': 0.0, 'coral': 0.0, 'kl': 0.0}
    progress_bar = tqdm(dataloader, desc=f"Phase 2 - Epoch {current_epoch}", unit="batch")
    
    for ecg, ppg, _ in progress_bar:
        ecg_target = ecg.to(device).float().unsqueeze(1)
        ppg_input = ppg.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        
        with autocast():
            # Lấy ECG latent không cần đạo hàm
            with torch.no_grad():
                z_ecg_1d, _ = model.encode_ecg(ecg_target)
            
            # Lấy PPG latent có đạo hàm
            z_ppg_1d, _ = model.encode_ppg(ppg_input)
            
            # Tính Distribution Alignment Loss (CORAL + KL)
            loss_dict = dist_align_fn(z_ppg_1d, z_ecg_1d, alpha_coral=ALPHA_CORAL, beta_kl=BETA_KL)
            total_loss = loss_dict['total_loss']

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.encode_ppg.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        metrics['total'] += total_loss.item()
        metrics['coral'] += loss_dict['coral_loss'].item()
        metrics['kl'] += loss_dict['kl_loss'].item()
        
        progress_bar.set_postfix({
            'Total': f"{total_loss.item():.4f}",
            'CORAL': f"{loss_dict['coral_loss'].item():.4f}",
            'KL': f"{loss_dict['kl_loss'].item():.4f}"
        })

    num_batches = len(dataloader)
    return {k: v / num_batches for k, v in metrics.items()}, z_ecg_1d, z_ppg_1d

def eval_phase2(model, dataloader, dist_align_fn, device):
    model.eval()
    t_total = 0.0
    with torch.no_grad():
        for ecg, ppg, _ in dataloader:
            ecg_target = ecg.to(device).float().unsqueeze(1)
            ppg_input = ppg.to(device).float().unsqueeze(1)
            
            z_ecg_1d, _ = model.encode_ecg(ecg_target)
            z_ppg_1d, _ = model.encode_ppg(ppg_input)
            
            loss_dict = dist_align_fn(z_ppg_1d, z_ecg_1d, alpha_coral=ALPHA_CORAL, beta_kl=BETA_KL)
            t_total += loss_dict['total_loss'].item()
            
    return t_total / len(dataloader)


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    set_seed(SEED)
    
    train_loader = DataLoader(LoadData('/home/linhhima/Diffusion datasets/train.npz'), 
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(LoadData('/home/linhhima/Diffusion datasets/val.npz'), 
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    model = ECG_PPG_Fusion_Model(embed_dim=OUTPUT_EMBED_DIM, freq_dim=256, target_length=SEQ_LENGTH).to(device)

    mse_fn = torch.nn.MSELoss().to(device)
    pearson_fn = PearsonLoss().to(device)                 
    
    # Khởi tạo Distribution Alignment Loss
    dist_align_fn = DistributionAlignmentLoss(eps=1e-8).to(device)
    
    scaler = GradScaler()

    # ---------------------------------------------------------
    # THỰC THI GIAI ĐOẠN 1
    # ---------------------------------------------------------
    if TRAIN_PHASE_1:
        print("\n" + "="*50)
        print("BẮT ĐẦU GIAI ĐOẠN 1: HUẤN LUYỆN ECG AUTOENCODER")
        print("="*50)
        
        # Optimizer GĐ1: Chỉ cập nhật encode_ecg và decoder
        optimizer_p1 = AdamW([
            {'params': model.encode_ecg.parameters(), 'lr': LR_CNN},
            {'params': model.decoder.parameters(), 'lr': LR_TRANSFORMER}
        ], weight_decay=1e-4)

        history_p1 = {'Train MSE': [], 'Test MSE': [], 'Train Pearson': []}
        best_test_mse = float('inf')

        for epoch in range(1, EPOCHS_PHASE1 + 1):
            avg_train = train_phase1(model, train_loader, optimizer_p1, mse_fn, pearson_fn, device, scaler, epoch)
            test_mse, test_corr = eval_phase1(model, test_loader, mse_fn, pearson_fn, device)
            
            history_p1['Train MSE'].append(avg_train['mse'])
            history_p1['Train Pearson'].append(avg_train['pearson'])
            history_p1['Test MSE'].append(test_mse)

            print(f"[Phase 1 Test] MSE: {test_mse:.6f} | Corr: {test_corr:.4f}")

            if test_mse < best_test_mse:
                best_test_mse = test_mse
                torch.save(model.state_dict(), PHASE1_CHECKPOINT)
                
            if epoch % 10 == 0:
                plot_losses(history_p1, "Phase 1: Reconstruction Loss", "phase1_history.png")
    else:
        print("\n[BỎ QUA] Huấn luyện Giai đoạn 1 theo cấu hình (TRAIN_PHASE_1 = False).")

    # ---------------------------------------------------------
    # THỰC THI GIAI ĐOẠN 2
    # ---------------------------------------------------------
    if TRAIN_PHASE_2:
        print("\n" + "="*50)
        print("BẮT ĐẦU GIAI ĐOẠN 2: CĂN CHỈNH PPG LATENT (DISTRIBUTION ALIGNMENT)")
        print("="*50)

        # CỰC KỲ QUAN TRỌNG: Nếu không chạy Giai đoạn 1, bắt buộc phải load trọng số cũ
        if not TRAIN_PHASE_1:
            print(f"Loading pre-trained Phase 1 weights from '{PHASE1_CHECKPOINT}'...")
            if os.path.exists(PHASE1_CHECKPOINT):
                model.load_state_dict(torch.load(PHASE1_CHECKPOINT, map_location=device))
                print("Tải trọng số Giai đoạn 1 thành công! Latent Space đã sẵn sàng.")
            else:
                print(f"LỖI NGHIÊM TRỌNG: Không tìm thấy file '{PHASE1_CHECKPOINT}'.")
                print("Bạn không thể train Giai đoạn 2 (PPG) nếu Giai đoạn 1 (ECG) chưa được cấu trúc đúng cách.")
                print("Vui lòng chạy lại với TRAIN_PHASE_1 = True hoặc cung cấp file trọng số hợp lệ.")
                exit(1)

        # Đóng băng (Freeze) toàn bộ trọng số của Giai đoạn 1
        for param in model.encode_ecg.parameters():
            param.requires_grad = False
        for param in model.decoder.parameters():
            param.requires_grad = False

        # Optimizer GĐ2: Chỉ cập nhật encode_ppg
        optimizer_p2 = AdamW(model.encode_ppg.parameters(), lr=LR_CNN, weight_decay=1e-4)
        
        # Cập nhật cấu trúc ghi lịch sử loss cho Phase 2
        history_p2 = {
            'Train Total Loss': [], 
            'Train CORAL Loss': [], 
            'Train KL Loss': [], 
            'Test Total Loss': []
        }
        best_test_loss = float('inf')

        for epoch in range(1, EPOCHS_PHASE2 + 1):
            train_metrics, last_emb_ecg, last_emb_ppg = train_phase2(
                model, train_loader, optimizer_p2, dist_align_fn, device, scaler, epoch
            )
            test_loss = eval_phase2(model, test_loader, dist_align_fn, device)
            
            history_p2['Train Total Loss'].append(train_metrics['total'])
            history_p2['Train CORAL Loss'].append(train_metrics['coral'])
            history_p2['Train KL Loss'].append(train_metrics['kl'])
            history_p2['Test Total Loss'].append(test_loss)

            print(f"[Phase 2 Test] Dist. Align Loss: {test_loss:.6f}")

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save(model.state_dict(), "best_phase2_fusion.pth")
                
                # Vẽ Similarity Matrix ở những epoch tốt nhất hoặc cuối cùng
                if epoch >= EPOCHS_PHASE2 - 10:
                    plot_best_similarity_matrix(last_emb_ecg, last_emb_ppg, epoch)

            if epoch % 10 == 0:
                plot_losses(history_p2, "Phase 2: Distribution Alignment Loss", "phase2_history.png")
    else:
        print("\n[BỎ QUA] Huấn luyện Giai đoạn 2 theo cấu hình (TRAIN_PHASE_2 = False).")

    print("\nChương trình thực thi kết thúc!")