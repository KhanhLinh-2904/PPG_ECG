import torch
import torch.nn as nn
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

from CLIP import PPG2ECGModel
from load_data import LoadData

# ==========================================
# CONSTANTS & HYPERPARAMETERS
# ==========================================
SEED = 44
NUM_EPOCHS = 500
PHASE_1_EPOCHS = 50      # Số epoch chỉ train Contrastive
BATCH_SIZE = 64
LR_CNN = 1e-4           
LR_TRANSFORMER = 5e-4    

# Trọng số cho Giai đoạn 2
WEIGHT_CONTRASTIVE = 1.0
WEIGHT_RECON = 1.0       # L1 Loss
WEIGHT_MSE = 1.0         # MSE Loss
WEIGHT_PEARSON = 1.0     # Pearson Loss

OUTPUT_EMBED_DIM = 128
SEQ_LENGTH = 2400

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRETRAINED_ECG_ENCODER = "/home/linhhima/PPG_ECG/Model6/best_ecg2ecg_autoencoder.pth" 

# ==========================================
# UTILITIES & LOSS FUNCTIONS
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

def plot_losses_combined(history, title='Multi-task Training Progress'):
    plt.figure(figsize=(12, 8))
    for label, values in history.items():
        if values:
            plt.plot(range(1, len(values) + 1), values, label=label)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True)
    plt.savefig("multitask_training_history.png")
    plt.close()

def pearson_loss(pred, target):
    """Tính toán (1 - Pearson Correlation)"""
    pred = pred.squeeze(1)     
    target = target.squeeze(1) 
    
    pred_mean = pred.mean(dim=1, keepdim=True)
    target_mean = target.mean(dim=1, keepdim=True)
    
    pred_centered = pred - pred_mean
    target_centered = target - target_mean
    
    cov = (pred_centered * target_centered).sum(dim=1)
    var_pred = (pred_centered ** 2).sum(dim=1).sqrt()
    var_target = (target_centered ** 2).sum(dim=1).sqrt()
    
    pearson = cov / (var_pred * var_target + 1e-8)
    return 1.0 - pearson.mean()

def contrastive_loss(z_ecg, z_ppg, temperature=0.07):
    """Tính InfoNCE Loss cho Contrastive Learning"""
    z_ecg = F.normalize(z_ecg, dim=-1)
    z_ppg = F.normalize(z_ppg, dim=-1)
    
    logits = (z_ppg @ z_ecg.t()) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    
    loss_p2e = F.cross_entropy(logits, labels)
    loss_e2p = F.cross_entropy(logits.t(), labels)
    return (loss_p2e + loss_e2p) / 2

# ==========================================
# CORE TRAINING FUNCTIONS
# ==========================================
def train_epoch(model, dataloader, optimizer, device, scaler, current_epoch):
    model.train()
    metrics = {k: 0.0 for k in ['total', 'contrastive', 'l1', 'mse', 'pearson']}
    
    # Xác định giai đoạn huấn luyện (Phase 1 hay Phase 2)
    is_phase_1 = current_epoch <= PHASE_1_EPOCHS
    phase_desc = "[Phase 1: Contrastive Only]" if is_phase_1 else "[Phase 2: Contrastive + Recon]"
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {current_epoch} {phase_desc}", unit="batch")
    last_emb_ecg, last_emb_ppg = None, None
    
    for ecg, ppg, _ in progress_bar:
        ecg_target = ecg.to(device).float().unsqueeze(1)
        ppg_input = ppg.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(ppg=ppg_input, ecg=ecg_target)
            
            z_ecg = outputs["z_ecg"]
            z_ppg = outputs["z_ppg"]
            recon_ecg = outputs.get("recon_ecg", None)
            
            last_emb_ecg = z_ecg
            last_emb_ppg = z_ppg
            
            # Tính Contrastive Loss (Luôn có)
            loss_cont = contrastive_loss(z_ecg, z_ppg)
            
            # Điều phối Loss theo Phase
            if is_phase_1:
                loss_l1 = torch.tensor(0.0, device=device)
                loss_mse = torch.tensor(0.0, device=device)
                loss_p = torch.tensor(0.0, device=device)
                total_loss = loss_cont
            else:
                loss_l1 = F.l1_loss(recon_ecg, ecg_target)
                loss_mse = F.mse_loss(recon_ecg, ecg_target)
                loss_p = pearson_loss(recon_ecg, ecg_target)
                total_loss = (WEIGHT_CONTRASTIVE * loss_cont) + (WEIGHT_RECON * loss_l1) + (WEIGHT_MSE * loss_mse) + (WEIGHT_PEARSON * loss_p)

        # Backpropagation
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        metrics['total'] += total_loss.item()
        metrics['contrastive'] += loss_cont.item()
        metrics['l1'] += loss_l1.item()
        metrics['mse'] += loss_mse.item()
        metrics['pearson'] += loss_p.item()

        postfix_dict = {'Total': f"{total_loss.item():.4f}", 'Cont': f"{loss_cont.item():.3f}"}
        if not is_phase_1:
            postfix_dict.update({'L1': f"{loss_l1.item():.4f}", 'MSE': f"{loss_mse.item():.4f}"})
        progress_bar.set_postfix(postfix_dict)

    num_batches = len(dataloader)
    return {k: v / num_batches for k, v in metrics.items()}, last_emb_ecg, last_emb_ppg


def evaluate(model, dataloader, device, current_epoch):
    model.eval()
    t_total, t_recon_l1, t_recon_mse, t_pearson, t_contrast = 0.0, 0.0, 0.0, 0.0, 0.0
    is_phase_1 = current_epoch <= PHASE_1_EPOCHS
    
    with torch.no_grad():
        for ecg, ppg, _ in dataloader:
            ecg_target = ecg.to(device).float().unsqueeze(1)
            ppg_input = ppg.to(device).float().unsqueeze(1)
            
            outputs = model(ppg=ppg_input, ecg=ecg_target)
            
            loss_cont = contrastive_loss(outputs["z_ecg"], outputs["z_ppg"])
            t_contrast += loss_cont.item()
            
            if not is_phase_1:
                recon_ecg = outputs["recon_ecg"]
                loss_l1 = F.l1_loss(recon_ecg, ecg_target)
                loss_mse = F.mse_loss(recon_ecg, ecg_target)
                loss_p = pearson_loss(recon_ecg, ecg_target)
                
                t_recon_l1 += loss_l1.item()
                t_recon_mse += loss_mse.item()
                t_pearson += loss_p.item()
                t_total += (WEIGHT_CONTRASTIVE * loss_cont + WEIGHT_RECON * loss_l1 + WEIGHT_MSE * loss_mse + WEIGHT_PEARSON * loss_p).item()
            else:
                t_total += loss_cont.item()
            
    n = len(dataloader)
    avg_corr = 1.0 - (t_pearson / n) if not is_phase_1 else 0.0
    
    return t_total / n, t_recon_l1 / n, t_recon_mse / n, avg_corr, t_contrast / n

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    set_seed(SEED)
    
    train_loader = DataLoader(LoadData('/home/linhhima/PPG_ECG/datasets/z_score_norm/mimic_III_train.npz'), 
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(LoadData('/home/linhhima/PPG_ECG/datasets/z_score_norm/mimic_III_test.npz'), 
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    model = PPG2ECGModel(proj_dim=OUTPUT_EMBED_DIM).to(device)

    # ---------------------------------------------------------
    # [QUAN TRỌNG] LOAD VÀ FREEZE DUY NHẤT ECG ENCODER
    # ---------------------------------------------------------
    print(f"Loading pre-trained ECG Encoder from {PRETRAINED_ECG_ENCODER}...")
    try:
        checkpoint = torch.load(PRETRAINED_ECG_ENCODER, map_location=device)
        model.ecg_enc.load_state_dict(checkpoint['encoder_state_dict'])
        
        # Chỉ Freeze khối ecg_enc (Ground Truth Encoder cho Contrastive Learning)
        for param in model.ecg_enc.parameters():
            param.requires_grad = False
        print("-> Successfully loaded and FROZEN the ECG Encoder.")
    except Exception as e:
        print(f"-> Error loading checkpoint: {e}. Please ensure the file exists and keys match.")
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # TẠO OPTIMIZER: Huấn luyện TOÀN BỘ các khối còn lại
    # ---------------------------------------------------------
    # Nhóm 1: Khối Attention (Fusion) - Dùng Learning Rate riêng của Transformer
    attention_params = [p for n, p in model.named_parameters() if "fusion" in n and p.requires_grad]
    
    # Nhóm 2: Các khối CNN và Linear còn lại (ppg_time_enc, ppg_freq_enc, decoder, proj_head)
    cnn_params = [p for n, p in model.named_parameters() if "fusion" not in n and p.requires_grad]

    optimizer = AdamW([
        {'params': cnn_params, 'lr': LR_CNN},              
        {'params': attention_params, 'lr': LR_TRANSFORMER} 
    ], weight_decay=1e-4)
    # ---------------------------------------------------------

    scaler = GradScaler()
    history = {'Total Loss': [], 'Contrastive Loss': [], 'Recon L1': [], 'Recon MSE': []}
    
    best_test_cont = float('inf')
    best_test_recon = float('inf')

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()

        avg_train, last_emb_ecg, last_emb_ppg = train_epoch(
            model, train_loader, optimizer, device, scaler, epoch
        )
        
        test_total, test_l1, test_mse, test_corr, test_cont = evaluate(
            model, test_loader, device, epoch
        )
        
        history['Total Loss'].append(avg_train['total'])
        history['Contrastive Loss'].append(avg_train['contrastive'])
        history['Recon L1'].append(avg_train['l1'])
        history['Recon MSE'].append(avg_train['mse'])

        print(f"Epoch {epoch} Summary (Time: {time.time() - start_time:.2f}s):")
        if epoch <= PHASE_1_EPOCHS:
            print(f"[Train] Contrastive: {avg_train['contrastive']:.4f}")
            print(f"[Test]  Contrastive: {test_cont:.4f}")
            
            # Phase 1: Save model dựa trên Contrastive Loss tốt nhất
            if test_cont < best_test_cont:
                best_test_cont = test_cont
                best_epoch = epoch
                torch.save(model.state_dict(), "best_multitask_phase1.pth")
                print(f"*** New Best Phase 1 Model Saved (Cont: {test_cont:.4f})")
                
        else:
            print(f"[Train] Total: {avg_train['total']:.4f} | Cont: {avg_train['contrastive']:.4f} | L1: {avg_train['l1']:.4f} | MSE: {avg_train['mse']:.4f}")
            print(f"[Test]  L1: {test_l1:.6f} | MSE: {test_mse:.6f} | Corr: {test_corr:.4f} | Cont: {test_cont:.4f}")
            
            # Phase 2: Save model dựa trên L1 Reconstruction tốt nhất
            if test_l1 < best_test_recon:
                best_test_recon = test_l1
                best_epoch = epoch
                best_emb_ecg = last_emb_ecg.detach().cpu().clone()
                best_emb_ppg = last_emb_ppg.detach().cpu().clone()
                torch.save(model.state_dict(), "best_multitask_phase2.pth")
                print(f"*** New Best Phase 2 Model Saved (L1: {test_l1:.6f})")

    print("Training Finished.")
    plot_losses_combined(history)
    
    if 'best_emb_ecg' in locals() and best_emb_ecg is not None:
        plot_best_similarity_matrix(best_emb_ecg, best_emb_ppg, best_epoch)