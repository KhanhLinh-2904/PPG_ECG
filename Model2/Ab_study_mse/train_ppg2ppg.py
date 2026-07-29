import os
import random
import numpy as np
from dataclasses import dataclass, asdict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt  

from load_data import LoadData
# Giả sử bạn lưu cấu trúc mô hình PPG AE ở file ppg2ppg.py
from ppg2ppg import PPGAutoencoder, PPGAEConfig, PPGReconstructionLoss

# ============================================================
# 1. Cấu hình Huấn luyện (Train Config)
# ============================================================
@dataclass
class TrainConfig:
    seed: int = 42

    train_path: str = "/home/linhhima/Diffusion_datasets/combined_train.npz"
    val_path: str = "/home/linhhima/Diffusion_datasets/combined_val.npz"

    save_dir: str = "/home/linhhima/PPG_ECG/Model2/Ab_study/saved_models_ppg_ae_subject"
    best_model_name: str = "best_ppg_autoencoder.pth"
    plot_name: str = "loss_curve_ppg_ppg_subject.png"  

    batch_size: int = 128
    epochs: int = 100
    num_workers: int = 4

    lr: float = 1e-4
    weight_decay: float = 1e-4

    grad_clip: float = 1.0
    use_amp: bool = True
    patience: int = 20  

    input_length: int = 2400
    ppg_in_channels: int = 1
    dims: tuple = (64, 128, 256, 512)
    depths: tuple = (2, 2, 4, 2)
    latent_channels: int = 16
    latent_length: int = 75
    attn_heads: int = 8
    attn_dropout: float = 0.0
    global_latent_dim: int = 128
    trend_poly: int = 2


CFG = TrainConfig()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(CFG.save_dir, exist_ok=True)

# ============================================================
# 2. Tiện ích bổ trợ (Utils)
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

def save_checkpoint(path, epoch, model, optimizer, best_loss, model_cfg, train_cfg):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_loss": best_loss,
        "model_config": model_cfg,
        "train_config": asdict(train_cfg),
    }, path)

def plot_loss_curve(train_losses, val_losses, save_dir, filename):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, label="Training Loss", color="blue", linewidth=2)
    plt.plot(epochs, val_losses, label="Validation Loss", color="red", linewidth=2, linestyle="--")
    
    plt.title("PPG Training and Validation Loss Curve", fontsize=14, fontweight='bold')
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Total Loss", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=12)
    
    plot_path = os.path.join(save_dir, filename)
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[+] Loss curve plotted and saved to: {plot_path}")

# ============================================================
# 3. Các hàm huấn luyện và đánh giá (Train / Eval)
# ============================================================
def train_one_epoch(model, criterion, optimizer, scaler, dataloader, epoch):
    model.train()
    total_loss_sum = 0.0

    loop = tqdm(dataloader, desc=f"Train Epoch {epoch}/{CFG.epochs}")
    for ppg, _, _ in loop:
        ppg = ppg.float().unsqueeze(1).to(DEVICE) if ppg.dim() == 2 else ppg.float().to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=CFG.use_amp):
            outputs = model(ppg)
            losses = criterion(outputs, ppg)
            loss = losses["loss_total"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss_sum += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss_sum / len(dataloader)

@torch.no_grad()
def validate_one_epoch(model, criterion, dataloader):
    model.eval()
    total_loss_sum = 0.0

    for ppg, _, _ in dataloader:
        ppg = ppg.float().unsqueeze(1).to(DEVICE) if ppg.dim() == 2 else ppg.float().to(DEVICE)

        outputs = model(ppg)
        losses = criterion(outputs, ppg)
        
        total_loss_sum += losses["loss_total"].item()

    return total_loss_sum / len(dataloader)

# ============================================================
# 4. Vòng lặp chính (Main Loop)
# ============================================================
def main():
    set_seed(CFG.seed)
    print(f"[*] Using device: {DEVICE}")

    train_loader, val_loader = get_dataloaders()

    model_cfg = PPGAEConfig(
        input_length=CFG.input_length, 
        ppg_in_channels=CFG.ppg_in_channels,
        dims=CFG.dims, 
        depths=CFG.depths, 
        latent_channels=CFG.latent_channels,
        latent_length=CFG.latent_length, 
        attn_heads=CFG.attn_heads,
        attn_dropout=CFG.attn_dropout, 
        global_latent_dim=CFG.global_latent_dim,
        trend_poly=CFG.trend_poly,
    )

    model = PPGAutoencoder(model_cfg).to(DEVICE)
    criterion = PPGReconstructionLoss().to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    
    # Đã sửa: Xóa bỏ tham số 'verbose=True' để tránh lỗi trên PyTorch mới
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=6
    )

    scaler = GradScaler(enabled=CFG.use_amp)

    best_val_loss = float("inf")
    bad_epochs = 0
    save_path = os.path.join(CFG.save_dir, CFG.best_model_name)

    train_losses = []
    val_losses = []

    print("\n[*] Starting PPG-to-PPG Autoencoder training process...")
    for epoch in range(1, CFG.epochs + 1):
        # Lưu lại Learning Rate cũ trước khi cập nhật epoch để theo dõi thủ công
        old_lr = optimizer.param_groups[0]['lr']

        # 1. Huấn luyện
        train_loss = train_one_epoch(model, criterion, optimizer, scaler, train_loader, epoch)
        
        # 2. Đánh giá (Validation)
        val_loss = validate_one_epoch(model, criterion, val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 3. Cập nhật Scheduler
        scheduler.step(val_loss)

        # Logic in log thủ công thay thế cho tham số verbose cũ
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < old_lr:
            print(f"   [i] Learning rate decreased from {old_lr:.6f} to {new_lr:.6f}")

        print(f" -> Results: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}")

        # 4. Lưu Checkpoint & Kích hoạt Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            bad_epochs = 0
            save_checkpoint(save_path, epoch, model, optimizer, best_val_loss, model_cfg, CFG)
            print(f"   [+] New best PPG model saved at epoch {epoch}!")
        else:
            bad_epochs += 1
            print(f"   [-] Loss did not improve for {bad_epochs} consecutive epoch(s).")

        if bad_epochs >= CFG.patience:
            print(f"\n[!] Early Stopping triggered: Halting early at epoch {epoch} due to no loss improvement.")
            break

    print("\n[*] Training complete successfully!")
    
    # 5. Vẽ đồ thị trực quan đường Loss
    plot_loss_curve(train_losses, val_losses, CFG.save_dir, CFG.plot_name)

if __name__ == "__main__":
    main()