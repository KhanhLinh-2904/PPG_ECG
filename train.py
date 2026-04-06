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
from torch.optim.lr_scheduler import CosineAnnealingLR
from CLIP import ECGEssembleCLIP, PPG_ECG_PatchGAN_Discriminator 
from loss import PearsonCorrelationLoss, self_clustering_contrastive_loss
from load_data import LoadData

# --- (CONSTANTS) ---
SEED = 44
NUM_EPOCHS = 250
LEARNING_RATE = 1e-4
BATCH_SIZE = 64

WEIGHT_CONTRASTIVE = 0.5
WEIGHT_L1 = 1.0      
WEIGHT_PEARSON = 2.0
WEIGHT_ADV = 0.1        

OUTPUT_EMBED_DIM = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def plot_best_similarity_matrix(embed_ecg, embed_ppg, epoch):
    raw_sim = (embed_ppg @ embed_ecg.t()).detach().cpu().numpy()
    
    plt.figure(figsize=(8, 6))
    plt.imshow(raw_sim, cmap='viridis', vmin=-1, vmax=1)
    plt.colorbar(label='Cosine Similarity')
    plt.title(f"PPG-ECG Similarity Matrix (Best Epoch: {epoch})\n")
    plt.xlabel("ECG Index (Ground Truth)")
    plt.ylabel("PPG Index (Predicted)")
    
    plt.savefig("best_similarity_matrix.png")
    plt.close()

def plot_losses_combined(losses_dict, title='Training Losses'):
    epochs = range(1, len(next(iter(losses_dict.values()))) + 1)
    plt.figure(figsize=(12, 8))
    
    for label, loss_values in losses_dict.items():
        if label != 'Total G Loss': 
            plt.plot(epochs, loss_values, label=label)
        
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{title.lower().replace(' ', '_')}.png")

def train_epoch_combined(
    model_G, model_D, dataloader, 
    optimizer_G, optimizer_D, 
    contrast_loss_fn, mse_loss_fn, pearson_loss_fn, adv_loss_fn,
    device, scaler_G, scaler_D, loss_weights
):
    model_G.train()
    model_D.train()
    
    metrics = {k: 0.0 for k in ['total_G', 'd_loss', 'contrast', 'mse', 'pearson', 'adv']}

    progress_bar = tqdm(dataloader, desc="Training Pix2Pix GAN", unit="batch")
    
    for ecg, ppg, _ in progress_bar:
        real_ecg = ecg.to(device).float().unsqueeze(1)
        ppg_input = ppg.to(device).float().unsqueeze(1)
        
        # =======================================================
        # 1. TRAIN DISCRIMINATOR (D)
        # =======================================================
        optimizer_D.zero_grad()
        with autocast():
            embed_ecg, embed_ppg, fake_ecg = model_G(real_ecg, ppg_input)
            
            pred_real = model_D(ppg_input, real_ecg)
            
            valid_labels = torch.ones_like(pred_real).to(device)
            fake_labels = torch.zeros_like(pred_real).to(device)
            
            d_loss_real = adv_loss_fn(pred_real, valid_labels)
            
            pred_fake = model_D(ppg_input, fake_ecg.detach())
            d_loss_fake = adv_loss_fn(pred_fake, fake_labels)
            d_loss = (d_loss_real + d_loss_fake) / 2
            
        scaler_D.scale(d_loss).backward()
        scaler_D.step(optimizer_D)
        scaler_D.update()
        
        # =======================================================
        # 2. TRAIN GENERATOR (G)
        # =======================================================
        optimizer_G.zero_grad()
        with autocast():
            pred_fake_for_G = model_D(ppg_input, fake_ecg)
            g_adv_loss = adv_loss_fn(pred_fake_for_G, valid_labels)
            
            c_loss = contrast_loss_fn(embed_ecg, embed_ppg)
            m_loss = mse_loss_fn(fake_ecg, real_ecg)
            p_loss = pearson_loss_fn(fake_ecg, real_ecg)

            g_loss = (loss_weights['contrast'] * c_loss + 
                      loss_weights['mse'] * m_loss + 
                      loss_weights['pearson'] * p_loss +
                      loss_weights['adv'] * g_adv_loss) 
        
        scaler_G.scale(g_loss).backward()
        scaler_G.step(optimizer_G)
        scaler_G.update()

        metrics['total_G'] += g_loss.item()
        metrics['d_loss'] += d_loss.item()
        metrics['contrast'] += c_loss.item() * loss_weights['contrast']
        metrics['mse'] += m_loss.item() * loss_weights['mse']
        metrics['pearson'] += p_loss.item() * loss_weights['pearson']
        metrics['adv'] += g_adv_loss.item() * loss_weights['adv']

        progress_bar.set_postfix({
            'Loss_G': f"{g_loss.item():.3f}", 
            'Loss_D': f"{d_loss.item():.3f}",
            'MSE': f"{m_loss.item():.3f}",
            'Adv_G': f"{g_adv_loss.item():.3f}"
        })

    num_batches = len(dataloader)
    avg_losses = {k: v / num_batches for k, v in metrics.items()}
    return avg_losses, embed_ecg, embed_ppg

if __name__ == "__main__":
    set_seed(SEED)
    print(f"Using device: {device}")

    train_dataset = LoadData('processed_data/mimic3_v1_2400_train.npz')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)

    model_G = ECGEssembleCLIP(embed_dim=OUTPUT_EMBED_DIM).to(device)
    model_D = PPG_ECG_PatchGAN_Discriminator(in_channels=2).to(device) 

    # Các hàm Loss
    mse_loss = nn.MSELoss().to(device) 
    pearson_loss = PearsonCorrelationLoss().to(device)
    adv_loss = nn.BCEWithLogitsLoss().to(device)
    

    optimizer_G = AdamW(model_G.parameters(), lr=1e-4, weight_decay=1e-4)
    optimizer_D = AdamW(model_D.parameters(), lr=5e-5, weight_decay=1e-4) 

    scheduler_G = CosineAnnealingLR(optimizer_G, T_max=NUM_EPOCHS, eta_min=1e-6)
    scheduler_D = CosineAnnealingLR(optimizer_D, T_max=NUM_EPOCHS, eta_min=1e-6)

    loss_weights = {
        'contrast': WEIGHT_CONTRASTIVE, 
        'mse': WEIGHT_L1, 
        'pearson': WEIGHT_PEARSON,
        'adv': WEIGHT_ADV
    }

    scaler_G = GradScaler()
    scaler_D = GradScaler()
    
    best_loss = float('inf')
    
    history = {
        'Total G Loss': [], 'Discriminator Loss': [], 'Contrastive Loss': [], 
        'MSE Loss': [], 'Pearson Loss': [], 'Adv G Loss': []
    }

    for epoch in range(1, NUM_EPOCHS + 1):
        start = time.time()

        avg_losses, last_embed_ecg, last_embed_ppg = train_epoch_combined(
            model_G, model_D, train_loader, 
            optimizer_G, optimizer_D,
            self_clustering_contrastive_loss, mse_loss, pearson_loss, adv_loss,
            device, scaler_G, scaler_D, loss_weights
        )
        scheduler_G.step()
        scheduler_D.step()
        current_lr_G = optimizer_G.param_groups[0]['lr']
        print(f"\nEpoch {epoch}/{NUM_EPOCHS} - Current LR (G): {current_lr_G:.6f}")

        history['Total G Loss'].append(avg_losses['total_G'])
        history['Discriminator Loss'].append(avg_losses['d_loss'])
        history['Contrastive Loss'].append(avg_losses['contrast'])
        history['MSE Loss'].append(avg_losses['mse'])
        history['Pearson Loss'].append(avg_losses['pearson'])
        history['Adv G Loss'].append(avg_losses['adv'])
        
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

        if avg_losses['total_G'] < best_loss:
            best_loss = avg_losses['total_G']
            torch.save(model_G.state_dict(), "multitask_best_model_G.pth")
            torch.save(model_D.state_dict(), "multitask_best_model_D.pth")
            print(f">> Saved best models at epoch {epoch} with Total G Loss: {best_loss:.4f}")
            plot_best_similarity_matrix(last_embed_ecg, last_embed_ppg, epoch)

        print(f"Epoch time: {time.time() - start:.1f}s")
        
    print("\nTraining Complete.")
    plot_losses_combined(history, title="Training Loss Components")