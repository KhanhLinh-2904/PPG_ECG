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
import torch.nn.functional as F
from utils import detect_ecg_features

# --- (CONSTANTS) ---
SEED = 44
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
WEIGHT_CONTRASTIVE = 0.1
WEIGHT_L1 = 1.0      
WEIGHT_PEARSON = 0.5 
WEIGHT_FREQUENCY = 0.01
WEIGHT_MASK = 1.0   
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

# --- PEARSON CORRELATION LOSS ---
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
    loss_weights
):
    model_clip.train()
    model_converter.train()

    metrics = {k: 0.0 for k in ['total', 'contrast', 'mse', 'pearson']}

    progress_bar = tqdm(dataloader, desc="Training (Multi-task)", unit="batch")
    
    for ecg, ppg, _  in progress_bar:
        ecg_target = ecg.to(device).float().unsqueeze(1)
        ppg_input = ppg.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        
        with autocast():
            logits_per_ecg, ppg_embedding, feature_lists_PPG = model_clip(ecg_target, ppg_input)
            c_loss = contrast_loss_fn(logits_per_ecg, ecg_target)
            predicted_ecg = model_converter(ppg_embedding, feature_lists_PPG) 

            m_loss = mse_loss_fn(predicted_ecg, ecg_target) 
            p_loss = pearson_loss_fn(predicted_ecg, ecg_target)
           


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
            'MSE': f"{m_loss.item():.4f}",
            'Pearson': f"{p_loss.item():.4f}",
        })

    num_batches = len(dataloader)
    avg_losses = {k: v / num_batches for k, v in metrics.items()}
    return avg_losses

if __name__ == "__main__":
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = LoadData('processed_data_single/record_30_train.npz')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                               shuffle=True, num_workers=4,
                                 pin_memory=True)


    model_clip = ECGEssembleCLIP(embed_dim=OUTPUT_EMBED_DIM).to(device)
    model_converter = ECGDecoder_UNet(target_length=2400).to(device)

    contrast_loss = FastSoftCLIPLoss(teacher_temp=0.05, student_temp=0.07).to(device)
    mse_loss = torch.nn.MSELoss().to(device) 
    pearson_loss = PearsonCorrelationLoss().to(device)
    
    params_to_optimize = (
                          list(model_clip.parameters()) + 
                          list(model_converter.parameters()))
    optimizer = AdamW(params_to_optimize, lr=LEARNING_RATE, weight_decay=1e-4)

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
            model_clip, model_converter, train_loader, optimizer,
            contrast_loss, mse_loss, pearson_loss, 
            device, scaler, loss_weights
        )
        
        history['Total Loss'].append(avg_losses['total'])
        history['Contrastive Loss'].append(avg_losses['contrast'])
        history['MSE Loss'].append(avg_losses['mse'])
        history['Pearson Loss'].append(avg_losses['pearson'])
        
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

        if avg_losses['total'] < best_loss:
            best_loss = avg_losses['total']
            torch.save(model_clip.state_dict(), "multitask_clip_best_model.pth")
            torch.save(model_converter.state_dict(), "multitask_decoder_best_model.pth")
            print(f">> Saved best model at epoch {epoch}")

        print(f"Epoch time: {time.time() - start:.1f}s")
        
    print("\nTraining Complete.")
    plot_losses_combined(history, title="Training Loss Components Over Epochs")