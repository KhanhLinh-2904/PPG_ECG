import os
import math
import random
import numpy as np
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import utility functions and model classes from Stage 1 & Stage 2
from load_data import LoadData
from ecg2ecg import ECGAutoencoder, ECGAEConfig
from ppg2ecg import PPG2ECGModel, PPG2ECGConfig
from flow_model import LatentRectifiedFlow

# ============================================================
# 1. CONFIGURATION FOR FLOW & REFLOW TRAINING
# ============================================================
@dataclass
class Stage2TrainConfig:
    seed: int = 42
    
    # Dataset file paths
    train_path: str = "/home/linhhima/Diffusion_datasets/combined_segment_split_train.npz"
    val_path: str = "/home/linhhima/Diffusion_datasets/combined_segment_split_val.npz"

    # Stage 1 Checkpoint Paths
    ecg_checkpoint_path: str = "/home/linhhima/PPG_ECG/PPG2ECG_RF/Proposed/saved_models_ecg_vae_segment/best_ecg_autoencoder.pth" 
    ppg_checkpoint_path: str = "/home/linhhima/PPG_ECG/PPG2ECG_RF/Proposed/saved_models_alignment_segment/best_ppg_alignment.pth"
    
    # Save directory and filenames for Stage 2 (Flow and Reflow)
    save_dir: str = "saved_models_reflow"
    best_model_name: str = "best_rectified_flow.pth"
    best_reflow_name: str = "best_2_rectified_flow.pth" # Checkpoint filename for Reflow model
    plot_name: str = "flow_loss_curve.png"  
    reflow_plot_name: str = "reflow_loss_curve.png"    # Filename for Reflow loss curve plot

    # General training parameters
    batch_size: int = 128  
    epochs: int = 300
    num_workers: int = 4
    lr: float = 2e-4       
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    use_amp: bool = True
    patience: int = 30

    # Reflow Euler Steps configuration for synthetic pair generation
    reflow_euler_steps: int = 20 # Number of ODE solver steps to generate z_new

    # Latent space architecture configuration
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

def plot_flow_losses(train_losses, val_losses, save_dir, filename, title_prefix="Latent Rectified Flow"):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="Train MSE", color="blue", linewidth=2)
    plt.plot(epochs, val_losses, label="Val MSE", color="red", linewidth=2, linestyle="--")
    plt.title(f"{title_prefix} - Training & Validation MSE", fontsize=14, fontweight="bold")
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("MSE Loss", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=12)
    plot_path = os.path.join(save_dir, filename)
    plt.savefig(plot_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[+] Saved loss curve plot successfully at: {plot_path}")

# ============================================================
# 4. LOAD FROZEN STAGE 1 MODELS
# ============================================================
def load_frozen_stage1():
    print("[*] Loading and freezing Stage 1 models (ECG Teacher & PPG Encoder)...")
    ecg_cfg = ECGAEConfig(
        input_length=CFG.input_length, in_channels=1, dims=CFG.dims, depths=CFG.depths,
        latent_channels=CFG.latent_channels, latent_length=CFG.latent_length,
        attn_heads=8, attn_dropout=0.0, global_latent_dim=128, trend_poly=2
    )
    ppg_cfg = PPG2ECGConfig(
        input_length=CFG.input_length, ppg_in_channels=1, dims=CFG.dims, depths=CFG.depths,
        latent_channels=CFG.latent_channels, latent_length=CFG.latent_length,
        attn_heads=8, attn_dropout=0.0, use_derivatives=True
    )
    torch.serialization.add_safe_globals([ECGAEConfig])
    ecg_ae = ECGAutoencoder(ecg_cfg).to(DEVICE)
    if not os.path.exists(CFG.ecg_checkpoint_path):
        raise FileNotFoundError(f"Error: ECG Teacher checkpoint not found at {CFG.ecg_checkpoint_path}")
    ckpt_ecg = torch.load(CFG.ecg_checkpoint_path, map_location=DEVICE)
    ecg_ae.load_state_dict(ckpt_ecg.get("model_state_dict", ckpt_ecg))
    ecg_ae.eval()
    for p in ecg_ae.parameters(): 
        p.requires_grad = False

    ppg_model = PPG2ECGModel(ecg_ae=ecg_ae, cfg=ppg_cfg).to(DEVICE)
    if not os.path.exists(CFG.ppg_checkpoint_path):
        raise FileNotFoundError(f"Error: PPG Alignment checkpoint not found at {CFG.ppg_checkpoint_path}")
    ckpt_ppg = torch.load(CFG.ppg_checkpoint_path, map_location=DEVICE)
    ppg_model.load_state_dict(ckpt_ppg.get("model_state_dict", ckpt_ppg))
    ppg_model.eval()
    for p in ppg_model.parameters(): 
        p.requires_grad = False

    return ecg_ae, ppg_model

# ============================================================
# 5. TRAINING & VALIDATION LOOPS FOR 1-RECTIFIED FLOW
# ============================================================
def train_one_epoch(flow_model, ecg_ae, ppg_model, optimizer, scaler, dataloader, epoch):
    flow_model.train()
    total_loss = 0.0
    loop = tqdm(dataloader, desc=f"Train Flow Epoch {epoch}/{CFG.epochs}")
    for ecg, ppg, _ in loop:
        ecg = ecg.float().unsqueeze(1).to(DEVICE) if ecg.dim() == 2 else ecg.float().to(DEVICE)
        ppg = ppg.float().unsqueeze(1).to(DEVICE) if ppg.dim() == 2 else ppg.float().to(DEVICE)
        B = ecg.size(0)

        with torch.no_grad():
            z_ecg = ecg_ae.latent_head(ecg_ae.encoder(ecg))
            z_ppg = ppg_model.ppg_latent_head(ppg_model.ppg_encoder(ppg))

        z0 = torch.randn_like(z_ecg)
        t = torch.rand((B,), device=DEVICE)
        t_expand = t.view(B, 1, 1)
        
        xt = t_expand * z_ecg + (1.0 - t_expand) * z0
        v_target = z_ecg - z0

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=CFG.use_amp):
            v_pred = flow_model(xt, t, z_ppg)
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

        z_ecg = ecg_ae.latent_head(ecg_ae.encoder(ecg))
        z_ppg = ppg_model.ppg_latent_head(ppg_model.ppg_encoder(ppg))

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
# 6. REFLOW UTILS: GENERATE STRAIGHTENED PAIRS
# ============================================================
@torch.no_grad()
def generate_reflow_dataset(flow_model, ecg_ae, ppg_model, dataloader, desc="Generating Reflow Data"):
    """Generates straightened pair datasets (z0, z_1) from the base Flow model via Euler ODE Solver."""
    flow_model.eval()
    all_z0 = []
    all_z1 = []
    all_z_ppg = []

    for ecg, ppg, _ in tqdm(dataloader, desc=desc):
        ecg = ecg.float().unsqueeze(1).to(DEVICE) if ecg.dim() == 2 else ecg.float().to(DEVICE)
        ppg = ppg.float().unsqueeze(1).to(DEVICE) if ppg.dim() == 2 else ppg.float().to(DEVICE)
        B = ecg.size(0)

        z_ecg = ecg_ae.latent_head(ecg_ae.encoder(ecg))
        z_ppg = ppg_model.ppg_latent_head(ppg_model.ppg_encoder(ppg))
        
        # Sample standardized Gaussian noise z0
        z0 = torch.randn_like(z_ecg)
        xt = z0.clone()
        
        # Setup Euler time-integration step to transport the trajectory
        dt = 1.0 / CFG.reflow_euler_steps
        for step in range(CFG.reflow_euler_steps):
            t_val = step * dt
            t_tensor = torch.full((B,), t_val, device=DEVICE, dtype=torch.float)
            
            # Predict velocity field at time t
            v_pred = flow_model(xt, t_tensor, z_ppg)
            # Euler step: update latent state
            xt = xt + v_pred * dt
            
        all_z0.append(z0.cpu())
        all_z1.append(xt.cpu()) # Straightened Target z_1
        all_z_ppg.append(z_ppg.cpu())

    # Package newly generated pairs into a TensorDataset
    return TensorDataset(
        torch.cat(all_z0, dim=0), 
        torch.cat(all_z1, dim=0), 
        torch.cat(all_z_ppg, dim=0)
    )

# ============================================================
# 7. TRAINING LOOP FOR 2-RECTIFIED FLOW (REFLOW)
# ============================================================
def train_reflow_epoch(reflow_model, optimizer, scaler, dataloader, epoch):
    reflow_model.train()
    total_loss = 0.0
    loop = tqdm(dataloader, desc=f"Train Reflow Epoch {epoch}/{CFG.epochs}")
    
    for z0, z1, z_ppg in loop:
        z0, z1, z_ppg = z0.to(DEVICE), z1.to(DEVICE), z_ppg.to(DEVICE)
        B = z0.size(0)
        
        t = torch.rand((B,), device=DEVICE)
        t_expand = t.view(B, 1, 1)
        
        # Straightened linear path: x_t = t * z1 + (1 - t) * z0
        xt = t_expand * z1 + (1.0 - t_expand) * z0
        v_target = z1 - z0 # Straightened target velocity vector field
        
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=CFG.use_amp):
            v_pred = reflow_model(xt, t, z_ppg)
            loss = F.mse_loss(v_pred, v_target)
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(reflow_model.parameters(), CFG.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        loop.set_postfix(Reflow_MSE=f"{loss.item():.5f}")
        
    return total_loss / len(dataloader)

@torch.no_grad()
def val_reflow_epoch(reflow_model, dataloader):
    reflow_model.eval()
    total_loss = 0.0
    for z0, z1, z_ppg in dataloader:
        z0, z1, z_ppg = z0.to(DEVICE), z1.to(DEVICE), z_ppg.to(DEVICE)
        B = z0.size(0)
        
        t = torch.rand((B,), device=DEVICE)
        t_expand = t.view(B, 1, 1)
        
        xt = t_expand * z1 + (1.0 - t_expand) * z0
        v_target = z1 - z0
        
        v_pred = reflow_model(xt, t, z_ppg)
        loss = F.mse_loss(v_pred, v_target)
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

# ============================================================
# 8. MAIN EXECUTION PIPELINE WITH REFLOW
# ============================================================
def main():
    set_seed(CFG.seed)
    print(f"[*] Running on device: {DEVICE}")

    # 1. Load Data
    train_loader, val_loader = get_dataloaders()

    # 2. Load Frozen Stage 1 Models
    ecg_ae, ppg_model = load_frozen_stage1()

    # 3. Initialize Base Flow Model (1-Rectified Flow)
    flow_model = LatentRectifiedFlow(
        latent_channels=CFG.latent_channels, 
        cond_channels=CFG.latent_channels, 
        hidden_dim=CFG.flow_hidden_dim, 
        num_blocks=CFG.flow_num_blocks
    ).to(DEVICE)

    # 4. Setup Optimizer & Scheduler for Stage 2 Flow
    optimizer = optim.AdamW(flow_model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    scaler = GradScaler(enabled=CFG.use_amp)

    best_val_loss = float("inf")
    bad_epochs = 0
    save_path = os.path.join(CFG.save_dir, CFG.best_model_name)

    train_losses = []
    val_losses = []

    # ============================================================
    # PHASE 1: TRAIN 1-RECTIFIED FLOW FIRST
    # ============================================================
    print("\n" + "="*55)
    print("[*] STARTING STAGE 2 TRAINING: 1-RECTIFIED FLOW")
    print("="*55 + "\n")

    for epoch in range(1, CFG.epochs + 1):
        train_mse = train_one_epoch(flow_model, ecg_ae, ppg_model, optimizer, scaler, train_loader, epoch)
        val_mse = val_one_epoch(flow_model, ecg_ae, ppg_model, val_loader)

        train_losses.append(train_mse)
        val_losses.append(val_mse)
        scheduler.step(val_mse)

        print(f"   => [Epoch {epoch}] Train MSE: {train_mse:.5f} | Val MSE: {val_mse:.5f}")

        if val_mse < best_val_loss:
            best_val_loss = val_mse
            bad_epochs = 0
            torch.save(flow_model.state_dict(), save_path)
            print(f"   [+] Saved best Flow model checkpoint (Val MSE decreased to {best_val_loss:.5f})!")
        else:
            bad_epochs += 1

        if bad_epochs >= CFG.patience:
            print(f"\n[!] Triggered Early Stopping for Flow model at epoch {epoch}.")
            break

    plot_flow_losses(train_losses, val_losses, CFG.save_dir, CFG.plot_name, title_prefix="1-Rectified Flow")

    # ============================================================
    # PHASE 2: REFLOW STAGE (2-RECTIFIED FLOW)
    # ============================================================
    print("\n" + "="*55)
    print("[*] STARTING REFLOW STAGE: GENERATING STRAIGHTENED DATASET & TRAINING 2-FLOW")
    print("="*55 + "\n")
    
    # Reload best 1-Flow model weights to generate reflow training pairs
    flow_model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    
    # Generate straightened dataset pairs (z0, z1) via ODE Solver
    reflow_train_set = generate_reflow_dataset(flow_model, ecg_ae, ppg_model, train_loader, desc="Generating Train Reflow Dataset")
    reflow_val_set = generate_reflow_dataset(flow_model, ecg_ae, ppg_model, val_loader, desc="Generating Val Reflow Dataset")
    
    reflow_train_loader = DataLoader(reflow_train_set, batch_size=CFG.batch_size, shuffle=True, pin_memory=True)
    reflow_val_loader = DataLoader(reflow_val_set, batch_size=CFG.batch_size, shuffle=False, pin_memory=True)
    
    # Initialize a new Reflow model (2-Rectified Flow)
    reflow_model = LatentRectifiedFlow(
        latent_channels=CFG.latent_channels, 
        cond_channels=CFG.latent_channels, 
        hidden_dim=CFG.flow_hidden_dim, 
        num_blocks=CFG.flow_num_blocks
    ).to(DEVICE)
    
    # Setup optimizer for Reflow
    reflow_optimizer = optim.AdamW(reflow_model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    reflow_scheduler = optim.lr_scheduler.ReduceLROnPlateau(reflow_optimizer, mode="min", factor=0.5, patience=10, verbose=True)
    
    best_reflow_loss = float("inf")
    reflow_bad_epochs = 0
    reflow_save_path = os.path.join(CFG.save_dir, CFG.best_reflow_name)
    
    reflow_train_losses = []
    reflow_val_losses = []
    
    for epoch in range(1, CFG.epochs + 1):
        r_train_mse = train_reflow_epoch(reflow_model, reflow_optimizer, scaler, reflow_train_loader, epoch)
        r_val_mse = val_reflow_epoch(reflow_model, reflow_val_loader)
        
        reflow_train_losses.append(r_train_mse)
        reflow_val_losses.append(r_val_mse)
        reflow_scheduler.step(r_val_mse)
        
        print(f"   => [Reflow Epoch {epoch}] Train MSE: {r_train_mse:.5f} | Val MSE: {r_val_mse:.5f}")
        
        if r_val_mse < best_reflow_loss:
            best_reflow_loss = r_val_mse
            reflow_bad_epochs = 0
            torch.save(reflow_model.state_dict(), reflow_save_path)
            print(f"   [+] Saved best REFLOW (2-Flow) model checkpoint (Val MSE reached {best_reflow_loss:.5f})!")
        else:
            reflow_bad_epochs += 1
            
        if reflow_bad_epochs >= CFG.patience:
            print(f"\n[!] Triggered Early Stopping for Reflow model at epoch {epoch}.")
            break
            
    # Render and save Reflow loss plot
    plot_flow_losses(reflow_train_losses, reflow_val_losses, CFG.save_dir, CFG.reflow_plot_name, title_prefix="2-Rectified Flow (Reflow)")
    print("\n[+] COMPLETED ENTIRE FLOW & REFLOW TRAINING PIPELINE SUCCESSFULLY!")

if __name__ == "__main__":
    main()