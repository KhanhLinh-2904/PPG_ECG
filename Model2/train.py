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

from CLIP import PPG2ECGModel
from loss import PPG2ECGLoss
from load_data import LoadData

# ==========================================
# CONSTANTS & HYPERPARAMETERS
# ==========================================
SEED = 44
NUM_EPOCHS = 200
BATCH_SIZE = 64
LR_CNN = 1e-4           
LR_TRANSFORMER = 5e-4    

WEIGHT_CONTRASTIVE = 1.0
WEIGHT_RECON = 1.0       
WEIGHT_PEARSON = 1.0
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
def train_epoch(model, dataloader, optimizer, criterion, device, scaler, current_epoch):
    model.train()
    metrics = {k: 0.0 for k in ['total', 'contrastive', 'recon', 'pearson']}
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {current_epoch} [Multi-task]", unit="batch")
    
    last_emb_ecg, last_emb_ppg = None, None
    
    for ecg, ppg, _ in progress_bar:
        ecg_target = ecg.to(device).float().unsqueeze(1)
        ppg_input = ppg.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(ppg=ppg_input, ecg=ecg_target)
            
            last_emb_ecg = outputs["z_ecg"]
            last_emb_ppg = outputs["z_ppg"]
            
            loss_dict = criterion(outputs, target_ecg=ecg_target)
            total_loss = loss_dict["loss_total"]

        # 3. Backpropagation  (Mixed Precision)
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
            
            # Forward
            outputs = model(ppg=ppg_input, ecg=ecg_target)
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
    
    # Load Dataloaders
    train_loader = DataLoader(LoadData('/home/linhhima/PPG_ECG/datasets/z_score_norm/mimic_III_train.npz'), 
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(LoadData('/home/linhhima/PPG_ECG/datasets/z_score_norm/mimic_III_test.npz'), 
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    model = PPG2ECGModel(proj_dim=OUTPUT_EMBED_DIM).to(device)

    criterion = PPG2ECGLoss(
        lambda_recon=WEIGHT_RECON, 
        lambda_pearson=WEIGHT_PEARSON, 
        lambda_contrastive=WEIGHT_CONTRASTIVE
    ).to(device)

    transformer_params = list(model.decoder.parameters())
    
    other_params = [p for n, p in model.named_parameters() if "decoder" not in n]
    other_params += list(criterion.parameters())

    optimizer = AdamW([
        {'params': other_params, 'lr': LR_CNN},
        {'params': transformer_params, 'lr': LR_TRANSFORMER}
    ], weight_decay=1e-4)

    scaler = GradScaler()
    history = {'Total Loss': [], 'Contrastive Loss': [], 'Recon Loss': [], 'Pearson Loss': []}
    best_test_recon = float('inf')

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()

        avg_train, last_emb_ecg, last_emb_ppg = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler, epoch
        )
        
        test_recon, test_corr, test_cont = evaluate(
            model, test_loader, criterion, device
        )
        
        history['Total Loss'].append(avg_train['total'])
        history['Contrastive Loss'].append(avg_train['contrastive'])
        history['Recon Loss'].append(avg_train['recon'])
        history['Pearson Loss'].append(avg_train['pearson'])

        print(f"Epoch {epoch} Summary (Time: {time.time() - start_time:.2f}s):")
        print(f"[Train] Total: {avg_train['total']:.4f} | Contrast: {avg_train['contrastive']:.4f}")
        print(f"[Test]  Recon (L1): {test_recon:.6f} | Corr: {test_corr:.4f} | Cont: {test_cont:.4f}")

        if test_recon < best_test_recon:
            best_test_recon = test_recon
            best_epoch = epoch
            
            best_emb_ecg = last_emb_ecg.detach().cpu().clone()
            best_emb_ppg = last_emb_ppg.detach().cpu().clone()
            
            torch.save(model.state_dict(), "best_multitask_model.pth")
            print(f"*** New Best Model Saved (Recon: {test_recon:.6f} at Epoch {epoch})")

    print("Training Finished.")

    plot_losses_combined(history)
    
    if best_emb_ecg is not None and best_emb_ppg is not None:
        plot_best_similarity_matrix(best_emb_ecg, best_emb_ppg, best_epoch)
        print(f"Saved similarity matrix for the best model (Epoch: {best_epoch}).")