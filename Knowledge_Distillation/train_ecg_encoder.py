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

from model import ECG2ECGModel 
from load_data import LoadData

# ==========================================
# CONSTANTS & HYPERPARAMETERS
# ==========================================
SEED = 44
NUM_EPOCHS = 200
BATCH_SIZE = 64
LR = 1e-4           

WEIGHT_RECON = 1.0       
WEIGHT_PEARSON = 1.0
SEQ_LENGTH = 2400

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# UTILITIES & LOSS FUNCTIONS
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def plot_losses_combined(history, title='ECG Autoencoder Training Progress'):
    plt.figure(figsize=(12, 8))
    for label, values in history.items():
        if values:
            plt.plot(range(1, len(values) + 1), values, label=label)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True)
    plt.savefig("ecg2ecg_training_history.png")
    plt.close()

def pearson_loss(pred, target):
    """Tính toán (1 - Pearson Correlation) để làm Loss function"""
    pred = pred.squeeze(1)     # (B, L)
    target = target.squeeze(1) # (B, L)
    
    pred_mean = pred.mean(dim=1, keepdim=True)
    target_mean = target.mean(dim=1, keepdim=True)
    
    pred_centered = pred - pred_mean
    target_centered = target - target_mean
    
    cov = (pred_centered * target_centered).sum(dim=1)
    var_pred = (pred_centered ** 2).sum(dim=1).sqrt()
    var_target = (target_centered ** 2).sum(dim=1).sqrt()
    
    pearson = cov / (var_pred * var_target + 1e-8)
    return 1.0 - pearson.mean()

# ==========================================
# CORE TRAINING FUNCTIONS
# ==========================================
def train_epoch(model, dataloader, optimizer, device, scaler, current_epoch):
    model.train()
    metrics = {k: 0.0 for k in ['total', 'recon', 'pearson']}
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {current_epoch} [Train]", unit="batch")
    
    for ecg, ppg, _ in progress_bar:
        ecg_target = ecg.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(ecg=ecg_target)
            recon_ecg = outputs["recon_ecg"]
            
            loss_l1 = F.l1_loss(recon_ecg, ecg_target)
            loss_p = pearson_loss(recon_ecg, ecg_target)
            
            total_loss = WEIGHT_RECON * loss_l1 + WEIGHT_PEARSON * loss_p

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        metrics['total'] += total_loss.item()
        metrics['recon'] += loss_l1.item()
        metrics['pearson'] += loss_p.item()

        progress_bar.set_postfix({
            'Total': f"{total_loss.item():.4f}", 
            'Recon(L1)': f"{loss_l1.item():.4f}",
            'Pearson': f"{loss_p.item():.4f}"
        })

    num_batches = len(dataloader)
    return {k: v / num_batches for k, v in metrics.items()}


def evaluate(model, dataloader, device):
    model.eval()
    t_total, t_recon, t_pearson = 0.0, 0.0, 0.0
    
    with torch.no_grad():
        for ecg, ppg, _ in dataloader:
            ecg_target = ecg.to(device).float().unsqueeze(1)
            
            outputs = model(ecg=ecg_target)
            recon_ecg = outputs["recon_ecg"]
            
            loss_l1 = F.l1_loss(recon_ecg, ecg_target)
            loss_p = pearson_loss(recon_ecg, ecg_target)
            total_loss = WEIGHT_RECON * loss_l1 + WEIGHT_PEARSON * loss_p
            
            t_total += total_loss.item()
            t_recon += loss_l1.item()
            t_pearson += loss_p.item()
            
    n = len(dataloader)
    avg_corr = 1.0 - (t_pearson / n) 
    
    return t_total / n, t_recon / n, avg_corr

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    set_seed(SEED)
    
    train_loader = DataLoader(LoadData('/home/linhhima/PPG_ECG/datasets/z_score_norm/mimic_III_train.npz'), 
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(LoadData('/home/linhhima/PPG_ECG/datasets/z_score_norm/mimic_III_test.npz'), 
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    model = ECG2ECGModel(in_channels=1).to(device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = GradScaler()
    
    history = {'Train Total Loss': [], 'Test Total Loss': [], 'Train Recon L1': [], 'Test Recon L1': []}
    best_test_recon = float('inf')

    print(f"Starting ECG2ECG Autoencoder Training on {device}...")

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()

        avg_train = train_epoch(model, train_loader, optimizer, device, scaler, epoch)
        test_total, test_recon, test_corr = evaluate(model, test_loader, device)
        
        history['Train Total Loss'].append(avg_train['total'])
        history['Test Total Loss'].append(test_total)
        history['Train Recon L1'].append(avg_train['recon'])
        history['Test Recon L1'].append(test_recon)

        print(f"Epoch {epoch} Summary (Time: {time.time() - start_time:.2f}s):")
        print(f"[Train] Total: {avg_train['total']:.4f} | Recon (L1): {avg_train['recon']:.4f} | Pearson Loss: {avg_train['pearson']:.4f}")
        print(f"[Test]  Total: {test_total:.4f} | Recon (L1): {test_recon:.6f} | Correlation: {test_corr:.4f}")

        if test_recon < best_test_recon:
            best_test_recon = test_recon
            best_epoch = epoch
            
            checkpoint = {
                'epoch': epoch,
                'encoder_state_dict': model.ecg_enc.state_dict(),
                'decoder_state_dict': model.decoder.state_dict(),
                'best_recon_l1_loss': best_test_recon,
                'best_correlation': test_corr
            }
            
            torch.save(checkpoint, "best_ecg2ecg_autoencoder.pth")
            print(f"*** New Best Model Saved (Test Recon L1: {test_recon:.6f} at Epoch {epoch})")
            print("    -> Encoder and Decoder weights saved separately.")

    print(f"Training Finished. Best Model was at Epoch {best_epoch}.")
    plot_losses_combined(history)