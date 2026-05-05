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

# Đảm bảo import đúng tên Class model mới
from CLIP import ECG_PPG_Fusion_Model 
from loss import PPG2ECGLoss
from load_data import LoadData

# ==========================================
# CONSTANTS & HYPERPARAMETERS
# ==========================================
SEED = 44
NUM_EPOCHS = 200
BATCH_SIZE = 64

# Base Learning Rates
LR_CNN = 1e-4           
LR_TRANSFORMER = 5e-4    

# Định nghĩa các mốc Epoch cho 3 giai đoạn
PHASE_1_END = 20
PHASE_2_END = 60
# Từ 61 trở đi sẽ là Phase 3

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
    embed_ecg = F.normalize(embed_ecg, dim=-1)
    embed_ppg = F.normalize(embed_ppg, dim=-1)
    
    raw_sim = (embed_ppg @ embed_ecg.t()).detach().cpu().numpy()
    plt.figure(figsize=(8, 6))
    plt.imshow(raw_sim, cmap='viridis', vmin=-1, vmax=1)
    plt.colorbar(label='Cosine Similarity')
    plt.title(f"PPG-ECG Similarity Matrix (Epoch: {epoch})")
    plt.xlabel("ECG Index")
    plt.ylabel("PPG Index")
    plt.savefig(f"similarity_matrix_epoch_{epoch}.png")
    plt.close()

def plot_losses_combined(history, title='Training Progress'):
    plt.figure(figsize=(12, 8))
    for label, values in history.items():
        if values:
            plt.plot(range(1, len(values) + 1), values, label=label)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True)
    plt.savefig("training_history.png")
    plt.close()

# ==========================================
# CORE TRAINING FUNCTIONS
# ==========================================
def train_epoch(model, dataloader, optimizer, criterion, device, scaler, current_epoch, phase_name):
    model.train()
    metrics = {k: 0.0 for k in ['total', 'contrastive', 'recon', 'pearson']}
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {current_epoch} [{phase_name}]", unit="batch")
    
    last_emb_ecg, last_emb_ppg = None, None
    
    for ecg, ppg, _ in progress_bar:
        ecg_target = ecg.to(device).float().unsqueeze(1)
        ppg_input = ppg.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(ecg_signal=ecg_target, ppg_signal=ppg_input)
            
            # Map key
            outputs["recon_ecg"] = outputs["reconstructed_ecg"]
            outputs["z_ecg"] = outputs["ecg_z"]
            outputs["z_ppg"] = outputs["ppg_z"]
            
            last_emb_ecg = outputs["ecg_z"]
            last_emb_ppg = outputs["ppg_z"]
            
            loss_dict = criterion(outputs, target_ecg=ecg_target)
            total_loss = loss_dict["loss_total"]

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        metrics['total'] += total_loss.item()
        metrics['contrastive'] += loss_dict.get("loss_contrastive", torch.tensor(0.0)).item()
        metrics['recon'] += loss_dict["loss_l1"].item()
        metrics['pearson'] += loss_dict["loss_pearson"].item()

        progress_bar.set_postfix({
            'Total': f"{total_loss.item():.4f}", 
            'Cont': f"{loss_dict.get('loss_contrastive', torch.tensor(0.0)).item():.3f}",
            'Recon': f"{loss_dict['loss_l1'].item():.4f}"
        })

    num_batches = len(dataloader)
    return {k: v / num_batches for k, v in metrics.items()}, last_emb_ecg, last_emb_ppg


def evaluate(model, dataloader, criterion, device):
    model.eval()
    t_recon, t_pearson, t_contrast = 0.0, 0.0, 0.0
    
    with torch.no_grad():
        for ecg, ppg, _ in dataloader:
            ecg_target = ecg.to(device).float().unsqueeze(1)
            ppg_input = ppg.to(device).float().unsqueeze(1)
            
            outputs = model(ecg_signal=ecg_target, ppg_signal=ppg_input)
            
            outputs["recon_ecg"] = outputs["reconstructed_ecg"]
            outputs["z_ecg"] = outputs.get("ecg_z")
            outputs["z_ppg"] = outputs.get("ppg_z")
                
            loss_dict = criterion(outputs, target_ecg=ecg_target)
            
            t_recon += loss_dict["loss_l1"].item()
            t_pearson += loss_dict["loss_pearson"].item()
            t_contrast += loss_dict.get("loss_contrastive", torch.tensor(0.0)).item()
            
    n = len(dataloader)
    avg_corr = 1.0 - (t_pearson / n) 
    
    return t_recon / n, avg_corr, t_contrast / n

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    set_seed(SEED)
    
    train_loader = DataLoader(LoadData('/home/linhhima/PPG_ECG/datasets/z_score_norm/mimic_III_train.npz'), 
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(LoadData('/home/linhhima/PPG_ECG/datasets/z_score_norm/mimic_III_test.npz'), 
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    model = ECG_PPG_Fusion_Model(embed_dim=OUTPUT_EMBED_DIM).to(device)

    # Khởi tạo loss với các giá trị mặc định (sẽ được cập nhật động trong loop)
    criterion = PPG2ECGLoss(
        lambda_contrastive=1.0,
        lambda_recon=0.0, 
        lambda_pearson=0.0
    ).to(device)

    transformer_params = list(model.decoder.parameters())
    other_params = [p for n, p in model.named_parameters() if "decoder" not in n]
    other_params += list(criterion.parameters())

    # Khởi tạo Optimizer (Index 0: Encoders, Index 1: Decoder)
    optimizer = AdamW([
        {'params': other_params, 'lr': LR_CNN},
        {'params': transformer_params, 'lr': LR_TRANSFORMER}
    ], weight_decay=1e-4)

    scaler = GradScaler()
    history = {'Total Loss': [], 'Contrastive Loss': [], 'Recon Loss': [], 'Pearson Loss': []}
    
    best_test_recon = float('inf')
    best_test_cont = float('inf')
    best_epoch = 0
    best_emb_ecg = None
    best_emb_ppg = None

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()
        
        # ==========================================
        # KIỂM SOÁT 3 GIAI ĐOẠN (3-PHASE TRAINING SCHEDULER)
        # ==========================================
        if epoch <= PHASE_1_END:
            phase_name = "Phase 1: Contrastive Warmup"
            # Đóng băng Decoder
            for param in model.decoder.parameters():
                param.requires_grad = False
            # Thiết lập trọng số
            criterion.lambda_contrastive = 1.0
            criterion.lambda_recon = 0.0
            criterion.lambda_pearson = 0.0

        elif PHASE_1_END < epoch <= PHASE_2_END:
            phase_name = "Phase 2: Decoder Warmup"
            # Mở khóa Decoder
            for param in model.decoder.parameters():
                param.requires_grad = True
            # Bật dần Reconstruction
            criterion.lambda_contrastive = 1.0
            criterion.lambda_recon = 0.2
            criterion.lambda_pearson = 0.2

        else:
            phase_name = "Phase 3: Joint Fine-tuning"
            # Đảm bảo Decoder đã mở khóa
            for param in model.decoder.parameters():
                param.requires_grad = True
            # Chuyển ưu tiên sang Reconstruction
            criterion.lambda_contrastive = 0.2
            criterion.lambda_recon = 1.0
            criterion.lambda_pearson = 1.0
            
            # Giảm Learning Rate khi vừa chuyển sang Phase 3 (chỉ làm ở Epoch 61)
            if epoch == PHASE_2_END + 1:
                print("\n>>> Phase 3 Started: Reducing Learning Rates by 5x! <<<")
                optimizer.param_groups[0]['lr'] = LR_CNN * 0.2        # Giảm LR cho encoder
                optimizer.param_groups[1]['lr'] = LR_TRANSFORMER * 0.2 # Giảm LR cho decoder
        
        # ==========================================

        # Gọi hàm train với phase_name để hiển thị lên thanh tqdm
        avg_train, last_emb_ecg, last_emb_ppg = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler, epoch, phase_name
        )
        
        test_recon, test_corr, test_cont = evaluate(
            model, test_loader, criterion, device
        )
        
        history['Total Loss'].append(avg_train['total'])
        history['Contrastive Loss'].append(avg_train['contrastive'])
        history['Recon Loss'].append(avg_train['recon'])
        history['Pearson Loss'].append(avg_train['pearson'])

        print(f"Epoch {epoch} Summary (Time: {time.time() - start_time:.2f}s):")
        print(f"[Train] Total: {avg_train['total']:.4f} | Cont: {avg_train['contrastive']:.4f} | Recon: {avg_train['recon']:.4f} | Corr: {1-avg_train['pearson']:.4f}")
        print(f"[Test]  Cont: {test_cont:.4f} | Recon (L1): {test_recon:.6f} | Corr: {test_corr:.4f}")

        # ==========================================
        # LƯU CHECKPOINT TÙY THEO GIAI ĐOẠN
        # ==========================================
        # Giai đoạn 1: Vì recon đang bị tắt, ta dựa vào Contrastive Loss để tìm model tốt nhất
        if epoch <= PHASE_1_END:
            if test_cont < best_test_cont:
                best_test_cont = test_cont
                torch.save(model.state_dict(), "best_phase1_contrastive.pth")
                print(f"*** Best Phase 1 Model Saved (Cont: {test_cont:.4f})")
                
        # Giai đoạn 2 & 3: Lưu dựa vào chất lượng sinh ECG (Recon Error)
        else:
            if test_recon < best_test_recon:
                best_test_recon = test_recon
                best_epoch = epoch
                
                best_emb_ecg = last_emb_ecg.detach().cpu().clone()
                best_emb_ppg = last_emb_ppg.detach().cpu().clone()
                
                torch.save(model.state_dict(), "best_multitask_model.pth")
                print(f"*** New Best Model Saved (Recon: {test_recon:.6f} at Epoch {epoch})")

    print("Training Finished. Generating final plots...")

    # plot_losses_combined(history)
    
    # if best_emb_ecg is not None and best_emb_ppg is not None:
    #     plot_best_similarity_matrix(best_emb_ecg, best_emb_ppg, best_epoch)
    #     print(f"Saved similarity matrix for the best model (Epoch: {best_epoch}).")