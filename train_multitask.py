from CLIP import ECGDecoder_UNet, ECGEssembleCLIP
from loss import FastSoftCLIPLoss
import torch
from torch.utils.data import DataLoader
from load_data import LoadData
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import numpy as np
import random

from utils import detect_ecg_features


# --- (CONSTANTS) ---
SEED = 44
INPUT_LENGTH = 2400
NUM_EPOCHS = 200
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
WEIGHT_CONTRASTIVE = 0.1
WEIGHT_L1 = 1.0      
WEIGHT_PEARSON = 0.5  
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
    plt.show()

# --- 1. PEARSON CORRELATION LOSS ---
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
        return 1 - torch.mean(r)

def train_epoch_combined(
    model_clip, model_converter, dataloader, optimizer, 
    contrast_loss_fn, mse_loss_fn, pearson_loss_fn, 
    device, scaler,
    loss_weights={'contrast': 1.0, 'mse': 1.0, 'pearson': 0.5, 'sample': 1.0} # Thêm tham số trọng số
):
    model_clip.train()
    model_converter.train()
    
    metrics = {k: 0.0 for k in ['total', 'contrast', 'mse', 'pearson', 'sample']}

    progress_bar = tqdm(dataloader, desc="Training (Multi-task)", unit="batch")
    
    for ecg, ppg, _ in progress_bar:
        ecg_target = ecg.to(device).float().unsqueeze(1)
        ppg_input = ppg.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()

        with autocast():
            # --- 1.  (Contrastive Task) ---
            logits_per_ecg, ppg_embedding, feature_lists_PPG = model_clip(ecg_target, ppg_input)
            c_loss = contrast_loss_fn(logits_per_ecg, ecg_target)
            
            # --- 2. (Reconstruction Task) ---
            predicted_ecg = model_converter(ppg_embedding, feature_lists_PPG) 

            #  MSE Loss
            m_loss = mse_loss_fn(predicted_ecg, ecg_target)

            # Pearson Loss
            p_loss = pearson_loss_fn(predicted_ecg, ecg_target)
            # Extrema Loss ---
            batch_size = ecg_target.shape[0]
            seq_len = ecg_target.shape[2]
            r_mask = torch.zeros_like(ecg_target) 
            o_mask = torch.zeros_like(ecg_target) 
            
    
            ecg_np = ecg_target.detach().cpu().numpy() 
            for i in range(batch_size):
                signal_i = ecg_np[i].flatten()
                r_idx, o_idx = detect_ecg_features(signal_i, frequency)
                if len(r_idx) > 0:
                    r_mask[i, 0, r_idx] = 1.0
                if len(o_idx) > 0:
                    o_mask[i, 0, o_idx] = 1.0

            r_mask = r_mask.to(device)
            o_mask = o_mask.to(device)

            r_count = torch.sum(r_mask).clamp(min=1)
            o_count = torch.sum(o_mask).clamp(min=1)
            
            loss_peak = torch.nn.functional.mse_loss(predicted_ecg * r_mask, ecg_target * r_mask, reduction='sum')
            loss_valley = torch.nn.functional.mse_loss(predicted_ecg * o_mask, ecg_target * o_mask, reduction='sum')
            
            # s_loss = (loss_peak / r_count) + (loss_valley / o_count)
            s_loss = (loss_peak / r_count) 

            total_loss = (loss_weights['contrast'] * c_loss + 
                          loss_weights['mse'] * m_loss + 
                          loss_weights['pearson'] * p_loss + 
                          loss_weights['sample'] * s_loss)

        
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        metrics['total'] += total_loss.item()
        metrics['contrast'] += c_loss.item() * loss_weights['contrast']
        metrics['mse'] += m_loss.item() * loss_weights['mse']
        metrics['pearson'] += p_loss.item() * loss_weights['pearson']
        metrics['sample'] += s_loss.item() * loss_weights['sample']

        progress_bar.set_postfix({
            'Total': f"{total_loss.item():.4f}", 
            'Con': f"{c_loss.item():.4f}",
            'MSE': f"{m_loss.item():.4f}",
            'Pears': f"{p_loss.item():.4f}",
            'Sample': f"{s_loss.item():.4f}"
        })

    torch.cuda.empty_cache()
    num_batches = len(dataloader)
    avg_losses = {k: v / num_batches for k, v in metrics.items()}
    return avg_losses['total'], avg_losses['contrast'], avg_losses['mse'], avg_losses['pearson'], avg_losses['sample']

if __name__ == "__main__":
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading training dataset...")
    train_dataset = LoadData('processed_data/mimic3_v1_train.npz') 
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    model_clip = ECGEssembleCLIP(embed_dim=OUTPUT_EMBED_DIM).to(device)
    model_converter = ECGDecoder_UNet(bottleneck_channels=2048).to(device)

    contrast_loss = FastSoftCLIPLoss(teacher_temp=0.05, student_temp=0.07).to(device)
    mse_loss = torch.nn.MSELoss().to(device)
    pearson_loss = PearsonCorrelationLoss().to(device)
    params_to_optimize = list(model_clip.parameters()) + list(model_converter.parameters())
    optimizer = AdamW(params_to_optimize, lr=LEARNING_RATE, weight_decay=1e-4)

    loss_weights = {
        'contrast': WEIGHT_CONTRASTIVE, 
        'mse': WEIGHT_L1, 
        'pearson': WEIGHT_PEARSON,
        'sample': 1.0  # Trọng số cho Extrema Loss (Peak/Valley)
    }

    scaler = GradScaler()
    best_loss = float('inf')
    
    history = {
        'Total Loss': [],
        'Contrastive Loss': [],
        'MSE Loss': [],
        'Pearson Loss': [],
        'Sample Loss': []
    }

    # ---- TRAINING ---
    for epoch in range(1, NUM_EPOCHS + 1):
        start = time.time()

        train_loss, train_contrast, train_mse, train_pearson, train_sample = train_epoch_combined(
            model_clip, model_converter, train_loader, optimizer,
            contrast_loss, mse_loss, pearson_loss, 
            device, scaler,
            loss_weights
        )
        
        history['Total Loss'].append(train_loss)
        history['Contrastive Loss'].append(train_contrast)
        history['MSE Loss'].append(train_mse)
        history['Pearson Loss'].append(train_pearson)
        history['Sample Loss'].append(train_sample)
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model_clip.state_dict(), "multitask_clip_best_model.pth")
            torch.save(model_converter.state_dict(), "multitask_decoder_best_model.pth")
            print(f">> Saved best model at epoch {epoch}")

        print(f"Epoch time: {time.time() - start:.1f}s")
        torch.cuda.empty_cache()
        
    print("\nTraining Complete.")
    
    plot_losses_combined(history, title="Training Loss Components Over Epochs")