import os
import math
import random
import numpy as np
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt  # Plotting library for loss curves

# Import Stage 1 & 2 utility functions and model architectures
from load_data import LoadData
from ecg2ecg import ECGAutoencoder, ECGAEConfig
from ppg2ecg import PPG2ECGModel, PPG2ECGConfig
from flow_model import LatentRectifiedFlow

# ============================================================
# 1. STAGE 2 CONFIGURATION
# ============================================================
@dataclass
class Stage2TrainConfig:
    seed: int = 42
    
    # Data paths
    train_path: str = "/home/linhhima/Diffusion_datasets/combined_segment_split_train.npz"
    val_path: str = "/home/linhhima/Diffusion_datasets/combined_segment_split_val.npz"

    # Stage 1 Checkpoint Paths (Requires completion of Stage 1 training)
    ecg_checkpoint_path: str = "/home/linhhima/PPG_ECG/PPG2ECG_RF/Proposed/saved_models_ecg_vae_segment/best_ecg_autoencoder.pth" 
    ppg_checkpoint_path: str = "/home/linhhima/PPG_ECG/PPG2ECG_RF/Pretrained_no_mse/saved_models_alignment_segment/best_ppg_alignment.pth"
    
    # Save directory and filenames for Stage 2
    save_dir: str = "/home/linhhima/PPG_ECG/PPG2ECG_RF/Pretrained_no_mse/saved_models_flow_segment"
    best_model_name: str = "best_rectified_flow.pth"
    plot_name: str = "flow_loss_curve.png"  # Loss curve plot filename

    # Training parameters for Rectified Flow
    batch_size: int = 128  # Maintained batch size of 128
    epochs: int = 300
    num_workers: int = 4
    lr: float = 2e-4       # Learning rate for Flow model
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    use_amp: bool = True
    patience: int = 30

    # Latent space architecture (Inherited from Stage 1)
    input_length: int = 2400
    latent_channels: int = 16
    latent_length: int = 75
    dims: tuple = (64, 128, 256, 512)
    depths: tuple = (2, 2, 4, 2)
    
    # Flow Model architecture parameters
    flow_hidden_dim: int = 128
    flow_num_blocks: int = 6

CFG = Stage2TrainConfig()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(CFG.save_dir, exist_ok=True)

# ============================================================
# 3. UTILITIES & DATA LOADERS
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_dataloaders():
    train_loader = DataLoader(
        LoadData(CFG.train_path), batch_size=CFG.batch_size, 
        shuffle=True, num_workers=CFG.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        LoadData(CFG.val_path), batch_size=CFG.batch_size, 
        shuffle=False, num_workers=CFG.num_workers, pin_memory=True
    )
    return train_loader, val_loader

# ---- LOSS CURVE PLOTTING UTILITY ----
def plot_flow_losses(train_losses, val_losses, save_dir, filename):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, label="Train MSE", color="blue", linewidth=2)
    plt.plot(epochs, val_losses, label="Val MSE", color="red", linewidth=2, linestyle="--")
    
    plt.title("Latent Rectified Flow - Training & Validation MSE", fontsize=14, fontweight="bold")
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("MSE Loss", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=12)
    
    plot_path = os.path.join(save_dir, filename)
    plt.savefig(plot_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"\n[+] Saved loss curve plot successfully at: {plot_path}")

# ============================================================
# 4. LOAD FROZEN STAGE 1 MODELS
# ============================================================
def load_frozen_stage1():
    print("[*] Loading and freezing Stage 1 models (ECG Teacher & PPG Encoder)...")
    
    # Initialize Configurations
    ecg_cfg = ECGAEConfig(
        input_length=CFG.input_length, in_channels=1, dims=CFG.dims, depths=CFG.depths,
        latent_channels=CFG.latent_channels, latent_length=CFG.latent_length,
        attn_heads=8, attn_dropout=0.0, global_latent_dim=128, trend_poly=2
    )
    # proj_dim removed to match updated PPG2ECGConfig
    ppg_cfg = PPG2ECGConfig(
        input_length=CFG.input_length, ppg_in_channels=1, dims=CFG.dims, depths=CFG.depths,
        latent_channels=CFG.latent_channels, latent_length=CFG.latent_length,
        attn_heads=8, attn_dropout=0.0, use_derivatives=True
    )

    # 1. Load ECG Teacher
    ecg_ae = ECGAutoencoder(ecg_cfg).to(DEVICE)
    if not os.path.exists(CFG.ecg_checkpoint_path):
        raise FileNotFoundError(f"Error: ECG Teacher checkpoint not found at {CFG.ecg_checkpoint_path}")
    ckpt_ecg = torch.load(CFG.ecg_checkpoint_path, map_location=DEVICE, weights_only=False)
    ecg_ae.load_state_dict(ckpt_ecg.get("model_state_dict", ckpt_ecg))
    ecg_ae.eval()
    for p in ecg_ae.parameters(): 
        p.requires_grad = False

    # 2. Load Aligned PPG Encoder
    ppg_model = PPG2ECGModel(ecg_ae=ecg_ae, cfg=ppg_cfg).to(DEVICE)
    if not os.path.exists(CFG.ppg_checkpoint_path):
        raise FileNotFoundError(f"Error: PPG Alignment checkpoint not found at {CFG.ppg_checkpoint_path}")
    ckpt_ppg = torch.load(CFG.ppg_checkpoint_path, map_location=DEVICE, weights_only=False)
    ppg_model.load_state_dict(ckpt_ppg.get("model_state_dict", ckpt_ppg))
    ppg_model.eval()
    for p in ppg_model.parameters(): 
        p.requires_grad = False

    return ecg_ae, ppg_model

# ============================================================
# 5. TRAINING & VALIDATION LOOPS
# ============================================================
def train_one_epoch(flow_model, ecg_ae, ppg_model, optimizer, scaler, dataloader, epoch):
    flow_model.train()
    total_loss = 0.0

    loop = tqdm(dataloader, desc=f"Train Epoch {epoch}/{CFG.epochs}")
    for ecg, ppg, _ in loop:
        ecg = ecg.float().unsqueeze(1).to(DEVICE) if ecg.dim() == 2 else ecg.float().to(DEVICE)
        ppg = ppg.float().unsqueeze(1).to(DEVICE) if ppg.dim() == 2 else ppg.float().to(DEVICE)
        B = ecg.size(0)

        # 1. EXTRACT LATENTS FROM FROZEN MODELS
        with torch.no_grad():
            feat_ecg = ecg_ae.encoder(ecg)
            z_ecg = ecg_ae.latent_head(feat_ecg)        # Target
            
            feat_ppg = ppg_model.ppg_encoder(ppg)
            z_ppg = ppg_model.ppg_latent_head(feat_ppg)  # Condition

        # 2. SETUP RECTIFIED FLOW
        z0 = torch.randn_like(z_ecg)  # Sample standard Gaussian noise N(0, I)
        
        # t ~ Uniform(0, 1)
        t = torch.rand((B,), device=DEVICE)
        t_expand = t.view(B, 1, 1)    # Shape: [B, 1, 1]
        
        # Linear interpolation path: x_t = t * z_ecg + (1 - t) * z0
        xt = t_expand * z_ecg + (1.0 - t_expand) * z0
        
        # Target vector field (Straight-line vector)
        v_target = z_ecg - z0

        # 3. BACKPROPAGATION & OPTIMIZATION
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=CFG.use_amp):
            # Predict vector field v
            v_pred = flow_model(xt, t, z_ppg)
            # MSE Loss Function
            loss = F.mse_loss(v_pred, v_target)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(flow_model.parameters(), CFG.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loop.set_postfix(MSE=f"{loss.item():.5f}")

    return total_loss / len(dataloader)

@torch.no_grad()
def val_one_epoch(flow_model, ecg_ae, ppg_model, dataloader):
    flow_model.eval()
    total_loss = 0.0

    for ecg, ppg, _ in dataloader:
        ecg = ecg.float().unsqueeze(1).to(DEVICE) if ecg.dim() == 2 else ecg.float().to(DEVICE)
        ppg = ppg.float().unsqueeze(1).to(DEVICE) if ppg.dim() == 2 else ppg.float().to(DEVICE)
        B = ecg.size(0)

        feat_ecg = ecg_ae.encoder(ecg)
        z_ecg = ecg_ae.latent_head(feat_ecg)

        feat_ppg = ppg_model.ppg_encoder(ppg)
        z_ppg = ppg_model.ppg_latent_head(feat_ppg)

        z0 = torch.randn_like(z_ecg)
        t = torch.rand((B,), device=DEVICE)
        t_expand = t.view(B, 1, 1)
        
        xt = t_expand * z_ecg + (1.0 - t_expand) * z0
        v_target = z_ecg - z0

        v_pred = flow_model(xt, t, z_ppg)
        loss = F.mse_loss(v_pred, v_target)

        total_loss += loss.item()

    return total_loss / len(dataloader)

# ============================================================
# 6. MAIN EXECUTION PIPELINE
# ============================================================
def main():
    set_seed(CFG.seed)
    print(f"[*] Running on device: {DEVICE}")

    # 1. Load Data
    train_loader, val_loader = get_dataloaders()

    # 2. Load Frozen Stage 1 Models
    ecg_ae, ppg_model = load_frozen_stage1()

    # 3. Initialize Stage 2 Flow Model
    flow_model = LatentRectifiedFlow(
        latent_channels=CFG.latent_channels, 
        cond_channels=CFG.latent_channels, 
        hidden_dim=CFG.flow_hidden_dim, 
        num_blocks=CFG.flow_num_blocks
    ).to(DEVICE)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(flow_model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    scaler = GradScaler(enabled=CFG.use_amp)

    best_val_loss = float("inf")
    bad_epochs = 0
    save_path = os.path.join(CFG.save_dir, CFG.best_model_name)

    # Initialize loss tracking lists across epochs
    train_losses = []
    val_losses = []

    print("\n" + "="*50)
    print("[*] STARTING STAGE 2 TRAINING: LATENT RECTIFIED FLOW")
    print("="*50 + "\n")

    for epoch in range(1, CFG.epochs + 1):
        
        train_mse = train_one_epoch(flow_model, ecg_ae, ppg_model, optimizer, scaler, train_loader, epoch)
        val_mse = val_one_epoch(flow_model, ecg_ae, ppg_model, val_loader)

        # Record losses to list
        train_losses.append(train_mse)
        val_losses.append(val_mse)

        scheduler.step(val_mse)

        print(f"   => [Epoch {epoch}] Train MSE: {train_mse:.5f} | Val MSE: {val_mse:.5f}")

        # Model Checkpointing
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            bad_epochs = 0
            torch.save(flow_model.state_dict(), save_path)
            print(f"   [+] Saved best Flow model checkpoint (Val MSE decreased to {best_val_loss:.5f})!")
        else:
            bad_epochs += 1
            print(f"   [-] Validation loss did not improve for {bad_epochs} epoch(s).")

        # Early Stopping check
        if bad_epochs >= CFG.patience:
            print(f"\n[!] Triggered Early Stopping at epoch {epoch}.")
            break

    # Generate and save loss curve plot upon completion
    plot_flow_losses(train_losses, val_losses, CFG.save_dir, CFG.plot_name)

if __name__ == "__main__":
    main()