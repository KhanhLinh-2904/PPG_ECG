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

def plot_losses_combined(history, title='Training vs Validation Progress'):
    pairs = [
        ('Train Total', 'Val Total'), 
        ('Train Contrastive', 'Val Contrastive'),
        ('Train MSE', 'Val MSE'), 
        ('Train Pearson', 'Val Pearson')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    for ax, (train_key, val_key) in zip(axes.flatten(), pairs):
        if history[train_key] and history[val_key]:
            epochs = range(1, len(history[train_key]) + 1)
            ax.plot(epochs, history[train_key], label=train_key, linestyle='-', linewidth=2)
            ax.plot(epochs, history[val_key], label=val_key, linestyle='--', linewidth=2)
            
            ax.set_title(train_key.replace('Train ', '') + ' Loss')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss Value')
            ax.legend()
            ax.grid(True)
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("training_validation_history.png")
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
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {current_epoch} [Train]", unit="batch")
    
    for ecg, ppg, _ in progress_bar:
        ecg_target = ecg.to(device).float().unsqueeze(1)
        ppg_input = ppg.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        
        with autocast():
            embed_ecg, embed_ppg, ecg_pred = model(ecg_target, ppg_input)
            
        c_loss = contrast_loss_fn(embed_ecg.float(), embed_ppg.float())
        m_loss = mse_loss_fn(ecg_pred.float(), ecg_target.float())
        p_loss = pearson_loss_fn(ecg_pred.float(), ecg_target.float())
        
        total_loss = (WEIGHT_CONTRASTIVE * c_loss + 
                      WEIGHT_MSE * m_loss + 
                      WEIGHT_PEARSON * p_loss)
        
        # 3. FIX: Chốt chặn an toàn chống nổ Gradient và hỏng Model
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("total_loss: ", total_loss)
            print('c_loss: ', c_loss)
            print('m_loss: ', m_loss)
            print('p_loss: ', p_loss)

            print(f"\n[CẢNH BÁO] Phát hiện NaN/Inf ở epoch {current_epoch}. Đã tự động bỏ qua batch này để bảo vệ mô hình!")
            optimizer.zero_grad()
            continue

        m_loss_val, p_loss_val = m_loss.item(), p_loss.item()

        # 4. Cập nhật Gradient
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
    metrics = {k: 0.0 for k in ['total', 'contrast', 'mse', 'pearson']}
    
    with torch.no_grad():
        for ecg, ppg, _ in dataloader:
            ecg_target = ecg.to(device).float().unsqueeze(1)
            ppg_input = ppg.to(device).float().unsqueeze(1)
            
            # Quá trình Eval không dùng autocast nên mặc định là float32, khá an toàn
            embed_ecg, embed_ppg, ecg_pred = model(ecg_target, ppg_input)
            
            c_loss = contrast_loss_fn(embed_ecg, embed_ppg)
            m_loss = mse_loss_fn(ecg_pred, ecg_target)
            p_loss = pearson_loss_fn(ecg_pred, ecg_target)
            
            total_loss = (WEIGHT_CONTRASTIVE * c_loss + 
                          WEIGHT_MSE * m_loss + 
                          WEIGHT_PEARSON * p_loss)
            
            # Đảm bảo không bị cộng NaN vào metrics nếu validation data có vấn đề
            if not (torch.isnan(total_loss) or torch.isinf(total_loss)):
                metrics['total'] += total_loss.item()
                metrics['contrast'] += c_loss.item()
                metrics['mse'] += m_loss.item()
                metrics['pearson'] += p_loss.item()
            
    n = len(dataloader)
    avg_metrics = {k: v / n for k, v in metrics.items()}
    avg_corr = 1.0 - avg_metrics['pearson'] 
    
    return avg_metrics, avg_corr

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    set_seed(SEED)
    
    train_loader = DataLoader(LoadData('/home/linhhima/Diffusion_datasets/combined_segment_split_train.npz'), 
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
                              
    val_loader = DataLoader(LoadData('/home/linhhima/Diffusion_datasets/combined_segment_split_val.npz'), 
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    model = ECG_PPG_Fusion_Model(embed_dim=OUTPUT_EMBED_DIM, freq_dim=256, target_length=SEQ_LENGTH).to(device)

    transformer_params = list(model.decoder.parameters())
    other_params = [p for n, p in model.named_parameters() if "decoder" not in n]

    optimizer = AdamW([
        {'params': other_params, 'lr': LR_CNN},
        {'params': transformer_params, 'lr': LR_TRANSFORMER}
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    mse_fn = torch.nn.MSELoss().to(device)
    pearson_fn = PearsonLoss().to(device)                  
    contrast_fn = ContrastiveLoss(temperature=0.1).to(device) 

    scaler = GradScaler()
    
    history = {
        'Train Total': [], 'Val Total': [],
        'Train Contrastive': [], 'Val Contrastive': [],
        'Train MSE': [], 'Val MSE': [],
        'Train Pearson': [], 'Val Pearson': []
    }
    
    best_val_mse = float('inf')

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()

        avg_train, last_emb_ecg, last_emb_ppg = train_epoch(
            model, train_loader, optimizer, contrast_fn, mse_fn, pearson_fn, device, scaler, epoch
        )
        
        val_metrics, val_corr = evaluate(
            model, val_loader, contrast_fn, mse_fn, pearson_fn, device
        )
        
        history['Train Total'].append(avg_train['total'])
        history['Val Total'].append(val_metrics['total'])
        
        history['Train Contrastive'].append(avg_train['contrast'])
        history['Val Contrastive'].append(val_metrics['contrast'])
        
        history['Train MSE'].append(avg_train['mse'])
        history['Val MSE'].append(val_metrics['mse'])
        
        history['Train Pearson'].append(avg_train['pearson'])
        history['Val Pearson'].append(val_metrics['pearson'])

        print(f"Epoch {epoch} Summary (Time: {time.time() - start_time:.2f}s):")
        print(f"[Train]      Total: {avg_train['total']:.4f} | MSE: {avg_train['mse']:.4f} | Cont: {avg_train['contrast']:.4f}")
        print(f"[Validation] Total: {val_metrics['total']:.4f} | MSE: {val_metrics['mse']:.6f} | Corr: {val_corr:.4f} | Cont: {val_metrics['contrast']:.4f}")

        if val_metrics['mse'] < best_val_mse:
            best_val_mse = val_metrics['mse']
            torch.save(model.state_dict(), "best_multitask_model.pth")
            print(f"*** New Best Model Saved (Validation MSE: {best_val_mse:.6f})")
            
            # Vẽ Similarity Matrix cho Best Model (chỉ thực hiện ở epoch cuối hoặc khi cần)
            if epoch == 199:
                plot_best_similarity_matrix(last_emb_ecg, last_emb_ppg, epoch)
                
        if epoch % 10 == 0:
            plot_losses_combined(history)
            
        # FIX: Scheduler theo dõi biến động của Validation MSE
        scheduler.step(val_metrics['mse'])

    print("Training Finished.")