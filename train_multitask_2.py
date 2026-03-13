import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import numpy as np
import random

from CLIP_2 import ECGEssembleCLIP
from loss import FastSoftCLIPLoss, PearsonCorrelationLoss, self_clustering_contrastive_loss
from load_data import LoadData

# --- (CONSTANTS) ---
SEED = 44
NUM_EPOCHS = 200
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
WEIGHT_CONTRASTIVE = 1.0
WEIGHT_L1 = 1.0      
WEIGHT_PEARSON = 0.5 
OUTPUT_EMBED_DIM = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def train_epoch_combined(
    model, dataloader, optimizer, 
    contrast_loss_fn, mse_loss_fn, pearson_loss_fn, 
    device, scaler, loss_weights
):
    model.train()
    metrics = {k: 0.0 for k in ['total', 'contrast', 'mse', 'pearson']}

    progress_bar = tqdm(dataloader, desc="Training (Multi-task)", unit="batch")
    
    for ecg, ppg, _ in progress_bar:
        ecg_target = ecg.to(device).float().unsqueeze(1)
        ppg_input = ppg.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        
        with autocast():
            # Bước Forward duy nhất: Mô hình trả về (ecg_embedding, ppg_embedding, ecg_reconstructed)
            embed_ecg, embed_ppg, ecg_pred = model(ecg_target, ppg_input)
            
            # Tính toán các thành phần Loss
            c_loss = contrast_loss_fn(embed_ecg, embed_ppg)
            m_loss = mse_loss_fn(ecg_pred, ecg_target)
            p_loss = pearson_loss_fn(ecg_pred, ecg_target)

            # Tổng hợp Loss
            total_loss = (loss_weights['contrast'] * c_loss + 
                          loss_weights['mse'] * m_loss + 
                          loss_weights['pearson'] * p_loss)
        
        # Cập nhật trọng số với Mixed Precision
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Lưu trữ metrics
        metrics['total'] += total_loss.item()
        metrics['contrast'] += c_loss.item() * loss_weights['contrast']
        metrics['mse'] += m_loss.item() * loss_weights['mse']
        metrics['pearson'] += p_loss.item() * loss_weights['pearson']

        progress_bar.set_postfix({
            'Total': f"{total_loss.item():.4f}", 
            'MSE': f"{m_loss.item():.4f}",
            'Pearson': f"{p_loss.item():.4f}",
        })

    num_batches = len(dataloader)
    avg_losses = {k: v / num_batches for k, v in metrics.items()}
    return avg_losses

if __name__ == "__main__":
    set_seed(SEED)
    print(f"Using device: {device}")

    # Load Data
    train_dataset = LoadData('processed_data/mimic3_v1_2400_train.npz')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)

    # Khởi tạo MỘT mô hình duy nhất bao gồm cả Encoder và Decoder
    model = ECGEssembleCLIP(embed_dim=OUTPUT_EMBED_DIM).to(device)

    # Khởi tạo loss functions
    mse_loss = torch.nn.MSELoss().to(device) 
    pearson_loss = PearsonCorrelationLoss().to(device)
   
    # Tối ưu hóa toàn bộ tham số trong ECGEssembleCLIP
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    loss_weights = {
        'contrast': WEIGHT_CONTRASTIVE, 
        'mse': WEIGHT_L1, 
        'pearson': WEIGHT_PEARSON
    }

    scaler = GradScaler()
    best_loss = float('inf')
    
    history = {
        'Total Loss': [], 'Contrastive Loss': [], 'MSE Loss': [], 
        'Pearson Loss': []
    }

    for epoch in range(1, NUM_EPOCHS + 1):
        start = time.time()

        avg_losses = train_epoch_combined(
            model, train_loader, optimizer,
            self_clustering_contrastive_loss, mse_loss, pearson_loss, 
            device, scaler, loss_weights
        )
        
        history['Total Loss'].append(avg_losses['total'])
        history['Contrastive Loss'].append(avg_losses['contrast'])
        history['MSE Loss'].append(avg_losses['mse'])
        history['Pearson Loss'].append(avg_losses['pearson'])
        
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

        # Lưu model dựa trên Total Loss
        if avg_losses['total'] < best_loss:
            best_loss = avg_losses['total']
            # Giờ chỉ cần lưu state_dict của model duy nhất
            torch.save(model.state_dict(), "multitask_best_model.pth")
            print(f">> Saved best model at epoch {epoch} with Total Loss: {best_loss:.4f}")

        print(f"Epoch time: {time.time() - start:.1f}s")
        
    print("\nTraining Complete.")
    plot_losses_combined(history, title="Training Loss Components")