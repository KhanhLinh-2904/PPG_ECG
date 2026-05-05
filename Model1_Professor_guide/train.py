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
from loss import ContrastiveLoss, PearsonLoss
from CLIP import ECG_PPG_Fusion_Model
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
WEIGHT_MSE = 1.0      
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
    embed_ecg = F.normalize(embed_ecg, dim=1)
    embed_ppg = F.normalize(embed_ppg, dim=1)
    
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
def train_epoch(
    model, dataloader, optimizer, 
    contrast_loss_fn, mse_loss_fn, pearson_loss_fn, 
    device, scaler, current_epoch
):
    model.train()
    metrics = {k: 0.0 for k in ['total', 'contrast', 'mse', 'pearson']}
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {current_epoch} [Multi-task]", unit="batch")
    
    for ecg, ppg, _ in progress_bar:
        ecg_target = ecg.to(device).float().unsqueeze(1)
        ppg_input = ppg.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        
        with autocast():
            embed_ecg, embed_ppg, ecg_pred = model(ecg_target, ppg_input)
            
            c_loss = contrast_loss_fn(embed_ecg, embed_ppg)
            m_loss = mse_loss_fn(ecg_pred, ecg_target)
            p_loss = pearson_loss_fn(ecg_pred, ecg_target)
            
            total_loss = (WEIGHT_CONTRASTIVE * c_loss + 
                          WEIGHT_MSE * m_loss + 
                          WEIGHT_PEARSON * p_loss)
            
            m_loss_val, p_loss_val = m_loss.item(), p_loss.item()

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        metrics['total'] += total_loss.item()
        metrics['contrast'] += c_loss.item()
        metrics['mse'] += m_loss_val
        metrics['pearson'] += p_loss_val

        progress_bar.set_postfix({
            'Total': f"{total_loss.item():.4f}", 
            'Contrast': f"{c_loss.item():.3f}",
            'MSE': f"{m_loss_val:.4f}"
        })

    num_batches = len(dataloader)
    return {k: v / num_batches for k, v in metrics.items()}, embed_ecg, embed_ppg

def evaluate(model, dataloader, contrast_loss_fn, mse_loss_fn, pearson_loss_fn, device):
    model.eval()
    t_mse, t_pearson, t_contrast = 0.0, 0.0, 0.0
    with torch.no_grad():
        for ecg, ppg, _ in dataloader:
            ecg_target = ecg.to(device).float().unsqueeze(1)
            ppg_input = ppg.to(device).float().unsqueeze(1)
            
            embed_ecg, embed_ppg, ecg_pred = model(ecg_target, ppg_input)
            
            t_contrast += contrast_loss_fn(embed_ecg, embed_ppg).item()
            t_mse += mse_loss_fn(ecg_pred, ecg_target).item()
            t_pearson += pearson_loss_fn(ecg_pred, ecg_target).item()
            
    n = len(dataloader)
    avg_corr = 1.0 - (t_pearson / n)
    return t_mse/n, avg_corr, t_contrast/n

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    set_seed(SEED)
    
    train_loader = DataLoader(LoadData('/home/linhhima/PPG_ECG/datasets/z_score_norm/mimic_III_train.npz'), 
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(LoadData('/home/linhhima/PPG_ECG/datasets/z_score_norm/mimic_III_test.npz'), 
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    model = ECG_PPG_Fusion_Model(embed_dim=OUTPUT_EMBED_DIM, freq_dim=256, target_length=SEQ_LENGTH).to(device)

    transformer_params = list(model.decoder.parameters())
    other_params = [p for n, p in model.named_parameters() if "decoder" not in n]

    optimizer = AdamW([
        {'params': other_params, 'lr': LR_CNN},
        {'params': transformer_params, 'lr': LR_TRANSFORMER}
    ], weight_decay=1e-4)

    mse_fn = torch.nn.MSELoss().to(device)
    pearson_fn = PearsonLoss().to(device)                 
    contrast_fn = ContrastiveLoss(temperature=0.1).to(device) 

    scaler = GradScaler()
    history = {'Total Loss': [], 'Contrastive Loss': [], 'MSE Loss': [], 'Pearson Loss': []}
    best_test_mse = float('inf')

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()

        avg_train, last_emb_ecg, last_emb_ppg = train_epoch(
            model, train_loader, optimizer, contrast_fn, mse_fn, pearson_fn, device, scaler, epoch
        )
        
        test_mse, test_corr, test_cont = evaluate(
            model, test_loader, contrast_fn, mse_fn, pearson_fn, device
        )
        
        history['Total Loss'].append(avg_train['total'])
        history['Contrastive Loss'].append(avg_train['contrast'])
        history['MSE Loss'].append(avg_train['mse'])
        history['Pearson Loss'].append(avg_train['pearson'])

        print(f"Epoch {epoch} Summary (Time: {time.time() - start_time:.2f}s):")
        print(f"[Train] Total: {avg_train['total']:.4f} | Contrast: {avg_train['contrast']:.4f}")
        print(f"[Test]  MSE: {test_mse:.6f} | Corr: {test_corr:.4f} | Cont: {test_cont:.4f}")

        if test_mse < best_test_mse:
            best_test_mse = test_mse
            torch.save(model.state_dict(), "best_multitask_model.pth")
            print(f"*** New Best Model Saved (MSE: {test_mse:.6f})")
            
            if epoch == 199:
                plot_best_similarity_matrix(last_emb_ecg, last_emb_ppg, epoch)

        if epoch % 10 == 0:
            plot_losses_combined(history)

    print("Training Finished.")