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

from CLIP import ECGEssembleCLIP
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
# freeze_encoder_ecg = "multitask_best_model_ecg.pth"
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
            embed_ecg, embed_ppg, ecg_pred = model(ecg_target, ppg_input)
            
            # ecg_pred = model(None, ppg_input)

            c_loss = contrast_loss_fn(embed_ecg, embed_ppg)
            m_loss = mse_loss_fn(ecg_pred, ecg_target)
            p_loss = pearson_loss_fn(ecg_pred, ecg_target)

            total_loss = (loss_weights['contrast'] * c_loss + 
                          loss_weights['mse'] * m_loss + 
                          loss_weights['pearson'] * p_loss)
            
            # total_loss = (
            #               loss_weights['mse'] * m_loss + 
            #               loss_weights['pearson'] * p_loss)
        
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

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
    return avg_losses, embed_ecg, embed_ppg
    # return avg_losses


if __name__ == "__main__":
    set_seed(SEED)
    print(f"Using device: {device}")

    train_dataset = LoadData('processed_data/mimic3_v1_2400_train.npz')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)

    model = ECGEssembleCLIP(embed_dim=OUTPUT_EMBED_DIM).to(device)

    # checkpoint = torch.load(freeze_encoder_ecg, map_location=device)
    # saved_state_dict = checkpoint.get('model_state_dict', checkpoint)
    

    # ecg_encoder_weights = {k.replace('encode_ecg.', ''): v for k, v in saved_state_dict.items() if k.startswith('encode_ecg.')}
    # if ecg_encoder_weights:
    #     model.encode_ecg.load_state_dict(ecg_encoder_weights, strict=True)  
    # for param in model.encode_ecg.parameters():
    #     param.requires_grad = False

    mse_loss = torch.nn.MSELoss().to(device) 
    pearson_loss = PearsonCorrelationLoss().to(device)
   
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

        avg_losses, last_embed_ecg, last_embed_ppg = train_epoch_combined(
            model, train_loader, optimizer,
            self_clustering_contrastive_loss, mse_loss, pearson_loss, 
            device, scaler, loss_weights
        )
        
        history['Total Loss'].append(avg_losses['total'])
        history['Contrastive Loss'].append(avg_losses['contrast'])
        history['MSE Loss'].append(avg_losses['mse'])
        history['Pearson Loss'].append(avg_losses['pearson'])
        
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

        if avg_losses['total'] < best_loss:
            best_loss = avg_losses['total']
            torch.save(model.state_dict(), "multitask_best_model.pth")
            print(f">> Saved best model at epoch {epoch} with Total Loss: {best_loss:.4f}")
            plot_best_similarity_matrix(last_embed_ecg, last_embed_ppg, epoch)

        print(f"Epoch time: {time.time() - start:.1f}s")
        
    print("\nTraining Complete.")
    plot_losses_combined(history, title="Training Loss Components")