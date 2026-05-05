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

from Knowledge_Distillation.Diffusion import ConditionNet, DiffusionUNetCrossAttention, ddpm_schedule
from load_data import LoadData

# ==========================================
# CONFIG & HYPERPARAMETERS
# ==========================================
CONFIG = {
    "seed": 42,
    "epochs": 400,
    "patience": 50,        # [MỚI] Số epoch tối đa chờ đợi nếu Test Loss không giảm
    "batch_size": 8,
    "nT": 200,             
    "beta1": 1e-4,         
    "beta2": 0.02,         
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "attention_heads": 4,
    "channels": 1,         
    "in_size": 2400,
    "num_workers": 4,
    "use_amp": True,
    "grad_clip": 1.0,
}

# ==========================================
# UTILITIES
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def plot_losses_combined(history, title='Training Progress'):
    plt.figure(figsize=(12, 8))
    for label, values in history.items():
        if values:
            plt.plot(range(1, len(values) + 1), values, label=label)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value (Noise MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig("diffusion_training_history.png")
    plt.close()

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# ==========================================
# CORE TRAINING FUNCTIONS
# ==========================================
def train_epoch(cond_net, unet, dataloader, optimizer, device, scaler, schedule_dict, current_epoch):
    cond_net.train()
    unet.train()
    
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {current_epoch} [Train]", unit="batch")
    
    sqrtab = schedule_dict["sqrtab"].to(device)
    sqrtmab = schedule_dict["sqrtmab"].to(device)
    nT = CONFIG["nT"]

    for ecg, ppg, _ in progress_bar:
        x_0 = ecg.to(device).float().unsqueeze(1)
        ppg_input = ppg.to(device).float().unsqueeze(1)
        batch_size = x_0.shape[0]
        
        t = torch.randint(1, nT + 1, (batch_size,), device=device).long()
        noise = torch.randn_like(x_0)
        
        sqrtab_t = extract(sqrtab, t, x_0.shape)
        sqrtmab_t = extract(sqrtmab, t, x_0.shape)
        x_t = sqrtab_t * x_0 + sqrtmab_t * noise
        
        optimizer.zero_grad()
        
        with autocast(enabled=CONFIG["use_amp"]):
            c = cond_net(ppg_input)
            noise_pred = unet(x_t, c, t.float())
            loss = F.mse_loss(noise_pred, noise)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(list(unet.parameters()) + list(cond_net.parameters()), max_norm=CONFIG["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        progress_bar.set_postfix({'Noise_MSE': f"{loss.item():.4f}"})

    return total_loss / len(dataloader)


def evaluate(cond_net, unet, dataloader, device, schedule_dict):
    cond_net.eval()
    unet.eval()
    
    total_loss = 0.0
    
    sqrtab = schedule_dict["sqrtab"].to(device)
    sqrtmab = schedule_dict["sqrtmab"].to(device)
    nT = CONFIG["nT"]
    
    with torch.no_grad():
        for ecg, ppg, _ in dataloader:
            x_0 = ecg.to(device).float().unsqueeze(1)
            ppg_input = ppg.to(device).float().unsqueeze(1)
            batch_size = x_0.shape[0]
            
            t = torch.randint(1, nT + 1, (batch_size,), device=device).long()
            noise = torch.randn_like(x_0)
            sqrtab_t = extract(sqrtab, t, x_0.shape)
            sqrtmab_t = extract(sqrtmab, t, x_0.shape)

            x_t = sqrtab_t * x_0 + sqrtmab_t * noise
            
            c = cond_net(ppg_input)
            noise_pred = unet(x_t, c, t.float())
            
            loss = F.mse_loss(noise_pred, noise)
            total_loss += loss.item()
            
    return total_loss / len(dataloader)

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    set_seed(CONFIG["seed"])
    device = CONFIG["device"]
    
    # 1. Load Dataloaders
    train_loader = DataLoader(LoadData('/home/linhhima/PPG_ECG/datasets/z_score_norm/mimic_III_train.npz'), 
                              batch_size=CONFIG["batch_size"], shuffle=True, 
                              num_workers=CONFIG["num_workers"], pin_memory=True)
    test_loader = DataLoader(LoadData('/home/linhhima/PPG_ECG/datasets/z_score_norm/mimic_III_test.npz'), 
                             batch_size=CONFIG["batch_size"], shuffle=False, 
                             num_workers=CONFIG["num_workers"], pin_memory=True)
    
    # 2. Initialize Diffusion Schedule
    schedule_dict = ddpm_schedule(beta1=CONFIG["beta1"], beta2=CONFIG["beta2"], T=CONFIG["nT"])
    
    # 3. Initialize the 2 component networks
    cond_net = ConditionNet().to(device)
    unet = DiffusionUNetCrossAttention(
        in_size=CONFIG["in_size"], 
        channels=CONFIG["channels"], 
        device=device, 
        num_heads=CONFIG["attention_heads"]
    ).to(device)

    # 4. Combine Parameters of both networks into 1 Optimizer
    optimizer = AdamW(
        list(cond_net.parameters()) + list(unet.parameters()), 
        lr=CONFIG["lr"], 
        weight_decay=CONFIG["weight_decay"]
    )
    
    scaler = GradScaler(enabled=CONFIG["use_amp"])
    
    history = {'Train Noise Loss': [], 'Test Noise Loss': []}
    
    best_test_loss = float('inf')
    epochs_no_improve = 0  # [MỚI] Biến đếm số epoch không cải thiện

    print(f"Starting to train the Conditional Diffusion model on device: {device}")

    # 5. Training Loop
    for epoch in range(1, CONFIG["epochs"] + 1):
        start_time = time.time()

        train_loss = train_epoch(cond_net, unet, train_loader, optimizer, device, scaler, schedule_dict, epoch)
        test_loss = evaluate(cond_net, unet, test_loader, device, schedule_dict)
        
        history['Train Noise Loss'].append(train_loss)
        history['Test Noise Loss'].append(test_loss)

        print(f"Epoch {epoch} Summary (Time: {time.time() - start_time:.2f}s):")
        print(f"[Train] Noise MSE: {train_loss:.6f}")
        print(f"[Test]  Noise MSE: {test_loss:.6f}")

        # [CẬP NHẬT] Logic Early Stopping
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            epochs_no_improve = 0  # Đặt lại bộ đếm về 0 nếu loss giảm
            torch.save({
                'cond_net_state_dict': cond_net.state_dict(),
                'unet_state_dict': unet.state_dict(),
            }, "best_conditional_diffusion.pth")
            print(f"*** New Best: Checkpoint saved successfully (Test Loss: {test_loss:.6f} at Epoch {epoch})")
        else:
            epochs_no_improve += 1
            print(f"--- No improvement for {epochs_no_improve} epoch(s).")
            
            if epochs_no_improve >= CONFIG["patience"]:
                print(f"\n[EARLY STOPPING] Kích hoạt tại Epoch {epoch}! Test loss không giảm trong {CONFIG['patience']} epochs liên tiếp.")
                break # Dừng vòng lặp huấn luyện

    print("Huấn luyện hoàn tất. Đang vẽ đồ thị...")
    plot_losses_combined(history, title='Diffusion Noise Prediction Loss')