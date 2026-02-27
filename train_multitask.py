from CLIP import ECGDecoder_UNet, ECGEssembleCLIP
from loss import FastSoftCLIPLoss
import torch
from torch.utils.data import DataLoader
from load_data import LoadData, PPG2ECG_Dataset, quality_aware_collate_fn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from tqdm import tqdm
import time
# import matplotlib as plt 
import matplotlib.pyplot as plt
import numpy as np
import random
import torch.nn.functional as F
from utils import detect_ecg_features

# --- (CONSTANTS) ---
SEED = 44
INPUT_LENGTH = 2048
NUM_EPOCHS = 200
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
WEIGHT_CONTRASTIVE = 0.1
WEIGHT_L1 = 1.0      
WEIGHT_PEARSON = 0.5 
WEIGHT_FREQUENCY = 0.01
OUTPUT_EMBED_DIM = 128
frequency = 125

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def plot_losses_combined(losses_dict, title='Training Losses'):
    epochs = range(1, len(next(iter(losses_dict.values()))) + 1)
    plt.figure(figsize=(10, 6))
    
    for label, loss_values in losses_dict.items():
        plt.plot(epochs, loss_values, label=label)
        
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{title.lower().replace(' ', '_')}.png")
    # plt.show() # Tắt show() nếu bạn chạy trên server không có màn hình

# --- 1. SỬA PEARSON CORRELATION LOSS ---
class PearsonCorrelationLoss(torch.nn.Module):
    def __init__(self):
        super(PearsonCorrelationLoss, self).__init__()

    def forward(self, x, y):
        x_flat = x.view(x.shape[0], -1)
        y_flat = y.view(y.shape[0], -1)
        
        mean_x = torch.mean(x_flat, dim=1, keepdim=True)
        mean_y = torch.mean(y_flat, dim=1, keepdim=True)
        
        xm = x_flat - mean_x
        ym = y_flat - mean_y
        
        r_num = torch.sum(xm * ym, dim=1)
        r_den = torch.sqrt(torch.sum(xm ** 2, dim=1) * torch.sum(ym ** 2, dim=1) + 1e-8)
        
        r = r_num / r_den
        # SỬA Ở ĐÂY: Trả về vector chứa Loss của từng batch thay vì gom lại bằng torch.mean()
        return 1 - r 
    
def frequency_loss_fn(pred, target):
    pred_fp32 = pred.to(torch.float32)
    target_fp32 = target.to(torch.float32)
    
    pred_fft = torch.fft.rfft(pred_fp32, dim=-1, norm='ortho')
    target_fft = torch.fft.rfft(target_fp32, dim=-1, norm='ortho')
    
    pred_mag = torch.abs(pred_fft)
    target_mag = torch.abs(target_fft)
    
    return torch.nn.functional.l1_loss(pred_mag, target_mag)

def train_epoch_combined(
    model_clip, model_converter, dataloader, optimizer, 
    contrast_loss_fn, mse_loss_fn, pearson_loss_fn, 
    device, scaler,
    loss_weights={'contrast': 1.0, 'mse': 1.0, 'pearson': 0.5}
):
    model_clip.train()
    model_converter.train()

    metrics = {k: 0.0 for k in ['total', 'contrast', 'mse', 'pearson']}

    progress_bar = tqdm(dataloader, desc="Training (Multi-task)", unit="batch")
    
    for ecg, ppg, _, ppg_sqi, ecg_sqi in progress_bar:
        ecg_target = ecg.to(device).float().unsqueeze(1)
        ppg_input = ppg.to(device).float().unsqueeze(1)
        
        # Shape: [Batch, 1, 1]
        ecg_sqi = ecg_sqi.to(device).float().view(-1, 1, 1)
        ppg_sqi = ppg_sqi.to(device).float().view(-1, 1, 1)
        
        optimizer.zero_grad()
        confidence_weights = ppg_sqi * ecg_sqi
        
        with autocast():
            # --- 1.  (Contrastive Task) ---
            logits_per_ecg, ppg_embedding, feature_lists_PPG = model_clip(ecg_target, ppg_input)
            c_loss = contrast_loss_fn(logits_per_ecg, ecg_target)
            
            # --- 2. (Reconstruction Task) ---
            predicted_ecg = model_converter(ppg_embedding, feature_lists_PPG) 

            unweighted_mse = mse_loss_fn(predicted_ecg, ecg_target) 
            m_loss = torch.mean(unweighted_mse * confidence_weights)

            unweighted_pearson = pearson_loss_fn(predicted_ecg, ecg_target)
            p_loss = torch.mean(unweighted_pearson.view(-1, 1, 1) * confidence_weights)

            total_loss = (loss_weights['contrast'] * c_loss + 
                          loss_weights['mse'] * m_loss + 
                          loss_weights['pearson'] * p_loss)
        
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        metrics['total'] += total_loss.item()
        metrics['contrast'] += c_loss.item() * loss_weights['contrast']
        metrics['mse'] += m_loss.item() * loss_weights['mse']
        metrics['pearson'] += p_loss.item() * loss_weights['pearson']

        progress_bar.set_postfix({
            'Total': f"{total_loss.item():.4f}", 
            'Con': f"{c_loss.item():.4f}",
            'MSE': f"{m_loss.item():.4f}",
            'Pears': f"{p_loss.item():.4f}",
        })

    num_batches = len(dataloader)
    avg_losses = {k: v / num_batches for k, v in metrics.items()}
    return avg_losses['total'], avg_losses['contrast'], avg_losses['mse'], avg_losses['pearson']

if __name__ == "__main__":
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading training dataset...")
    train_dataset = PPG2ECG_Dataset('datasets/total_z_train.npz') 
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                               shuffle=True, num_workers=4,
                                 pin_memory=True, collate_fn=quality_aware_collate_fn)

    model_clip = ECGEssembleCLIP(embed_dim=OUTPUT_EMBED_DIM).to(device)
    model_converter = ECGDecoder_UNet(bottleneck_channels=2048).to(device)

    contrast_loss = FastSoftCLIPLoss(teacher_temp=0.05, student_temp=0.07).to(device)
    
    # --- SỬA Ở ĐÂY: Thêm reduction='none' để lấy sai số của từng điểm ---
    mse_loss = torch.nn.MSELoss(reduction='none').to(device) 
    
    pearson_loss = PearsonCorrelationLoss().to(device)
    params_to_optimize = list(model_clip.parameters()) + list(model_converter.parameters())
    optimizer = AdamW(params_to_optimize, lr=LEARNING_RATE, weight_decay=1e-4)

    loss_weights = {
        'contrast': WEIGHT_CONTRASTIVE, 
        'mse': WEIGHT_L1, 
        'pearson': WEIGHT_PEARSON,
    }

    scaler = GradScaler()
    best_loss = float('inf')
    
    history = {
        'Total Loss': [],
        'Contrastive Loss': [],
        'MSE Loss': [],
        'Pearson Loss': [],
    }

    # ---- TRAINING ---
    for epoch in range(1, NUM_EPOCHS + 1):
        start = time.time()

        train_loss, train_contrast, train_mse, train_pearson = train_epoch_combined(
            model_clip, model_converter, train_loader, optimizer,
            contrast_loss, mse_loss, pearson_loss, 
            device, scaler,
            loss_weights
        )
        
        history['Total Loss'].append(train_loss)
        history['Contrastive Loss'].append(train_contrast)
        history['MSE Loss'].append(train_mse)
        history['Pearson Loss'].append(train_pearson)
        
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model_clip.state_dict(), "multitask_clip_best_model.pth")
            torch.save(model_converter.state_dict(), "multitask_decoder_best_model.pth")
            print(f">> Saved best model at epoch {epoch}")

        print(f"Epoch time: {time.time() - start:.1f}s")
        
    print("\nTraining Complete.")
    
    plot_losses_combined(history, title="Training Loss Components Over Epochs")