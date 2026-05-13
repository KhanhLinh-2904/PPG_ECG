import os
import random
import numpy as np
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from load_data import LoadData
from ecg2ecg import ECGAutoencoder, ECGAEConfig
from ppg2ecg import PPG2ECGModel, PPG2ECGConfig, CardioAlignLoss

# ============================================================
# 1. CẤU HÌNH (CONFIG)
# ============================================================
@dataclass
class TrainConfig:
    seed: int = 42
    
    train_path: str = "/home/linhhima/Diffusion datasets/train.npz"
    val_path: str = "/home/linhhima/Diffusion datasets/val.npz"

    ecg_checkpoint_path: str = "saved_models_ecg_vae/best_ecg_autoencoder.pth" 
    save_dir: str = "saved_models_alignment"
    best_model_name: str = "best_ppg_alignment.pth"

    # Training Params
    batch_size: int = 256
    epochs: int = 200
    num_workers: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    use_amp: bool = True
    patience: int = 20

    # Model ECG Params
    input_length: int = 2400
    in_channels: int = 1
    dims: tuple = (64, 128, 256, 512)
    depths: tuple = (2, 2, 4, 2)
    latent_channels: int = 16
    latent_length: int = 75
    attn_heads: int = 8
    attn_dropout: float = 0.0
    global_latent_dim: int = 128
    trend_poly: int = 2

    # Model PPG Params
    use_derivatives: bool = True
    proj_dim: int = 128

    # --- TRỌNG SỐ CHO CÁC HÀM LOSS ---
    coral_weight: float = 1.0
    kl_weight: float = 1.0
    smoothl1_weight: float = 0.0  # Trọng số cho Pairwise Smooth L1 Loss

CFG = TrainConfig()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(CFG.save_dir, exist_ok=True)

# ============================================================
# 2. UTILS & DATA
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

# ============================================================
# 3. BUILD TEACHER MODEL (FROZEN)
# ============================================================
def build_frozen_ecg_teacher():
    print(f"[*] Đang tải mô hình ECG Teacher từ: {CFG.ecg_checkpoint_path}")
    ecg_cfg = ECGAEConfig(
        input_length=CFG.input_length,
        in_channels=CFG.in_channels,
        dims=CFG.dims,
        depths=CFG.depths,
        latent_channels=CFG.latent_channels,
        latent_length=CFG.latent_length,
        attn_heads=CFG.attn_heads,
        attn_dropout=CFG.attn_dropout,
        global_latent_dim=CFG.global_latent_dim,
        trend_poly=CFG.trend_poly,
    )

    ecg_ae = ECGAutoencoder(ecg_cfg).to(DEVICE)
    
    checkpoint = torch.load(CFG.ecg_checkpoint_path, map_location=DEVICE)
    if "model_state_dict" in checkpoint:
        ecg_ae.load_state_dict(checkpoint["model_state_dict"])
    else:
        ecg_ae.load_state_dict(checkpoint)

    ecg_ae.eval()
    for p in ecg_ae.parameters():
        p.requires_grad = False
        
    return ecg_ae

# ============================================================
# 4. TRAINING & VALIDATION LOOPS
# ============================================================
def train_one_epoch(model, criterion, optimizer, scaler, dataloader, epoch):
    model.train()
    
    # Thêm 'pair' để track Smooth L1 Loss
    running = {"total": 0.0, "coral": 0.0, "kl": 0.0, "pair": 0.0}

    loop = tqdm(dataloader, desc=f"Train Epoch {epoch}/{CFG.epochs}")
    for ecg, ppg, _ in loop:
        ecg = ecg.float().unsqueeze(1).to(DEVICE) if ecg.dim() == 2 else ecg.float().to(DEVICE)
        ppg = ppg.float().unsqueeze(1).to(DEVICE) if ppg.dim() == 2 else ppg.float().to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=CFG.use_amp):
            outputs = model(ppg=ppg, ecg=ecg)
            
            # Tính Alignment Loss (CORAL + KL + Smooth L1)
            align_losses = criterion(outputs)
            loss_total = align_losses["loss_total"]

        scaler.scale(loss_total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        running["total"] += loss_total.item()
        running["coral"] += align_losses["loss_coral"].item()
        running["kl"] += align_losses["loss_kl"].item()
        running["pair"] += align_losses["loss_pair"].item()  # Cộng dồn Pair Loss

        loop.set_postfix(
            Tot=f"{loss_total.item():.3f}",
            Coral=f"{align_losses['loss_coral'].item():.4f}",
            KL=f"{align_losses['loss_kl'].item():.3f}",
            Pair=f"{align_losses['loss_pair'].item():.4f}",  # Hiển thị trên thanh tiến trình
        )

    n = len(dataloader)
    return {k: v / n for k, v in running.items()}


@torch.no_grad()
def validate_one_epoch(model, criterion, dataloader):
    model.eval()
    running = {"total": 0.0, "coral": 0.0, "kl": 0.0, "pair": 0.0}

    for ecg, ppg, _ in dataloader:
        ecg = ecg.float().unsqueeze(1).to(DEVICE) if ecg.dim() == 2 else ecg.float().to(DEVICE)
        ppg = ppg.float().unsqueeze(1).to(DEVICE) if ppg.dim() == 2 else ppg.float().to(DEVICE)

        outputs = model(ppg=ppg, ecg=ecg)
        align_losses = criterion(outputs)
        
        loss_total = align_losses["loss_total"]

        running["total"] += loss_total.item()
        running["coral"] += align_losses["loss_coral"].item()
        running["kl"] += align_losses["loss_kl"].item()
        running["pair"] += align_losses["loss_pair"].item()

    n = len(dataloader)
    return {k: v / n for k, v in running.items()}


# ============================================================
# 5. MAIN
# ============================================================
def main():
    set_seed(CFG.seed)
    print(f"[*] Đang sử dụng thiết bị: {DEVICE}")

    train_loader, val_loader = get_dataloaders()

    # 1. Build Frozen ECG Teacher
    ecg_teacher = build_frozen_ecg_teacher()

    # 2. Cấu hình & Build PPG Student
    ppg_cfg = PPG2ECGConfig(
        input_length=CFG.input_length,
        ppg_in_channels=1,
        dims=CFG.dims,
        depths=CFG.depths,
        latent_channels=CFG.latent_channels,
        latent_length=CFG.latent_length,
        attn_heads=CFG.attn_heads,
        attn_dropout=CFG.attn_dropout,
        use_derivatives=CFG.use_derivatives,
        proj_dim=CFG.proj_dim,
    )

    model = PPG2ECGModel(ecg_ae=ecg_teacher, cfg=ppg_cfg).to(DEVICE)

    # 3. Khởi tạo hàm Loss với các trọng số từ CFG
    criterion = CardioAlignLoss(
        coral_weight=CFG.coral_weight, 
        kl_weight=CFG.kl_weight,
        smoothl1_weight=CFG.smoothl1_weight
    ).to(DEVICE)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )
    scaler = GradScaler(enabled=CFG.use_amp)

    best_val_loss = float("inf")
    bad_epochs = 0
    save_path = os.path.join(CFG.save_dir, CFG.best_model_name)

    print("\n[*] Bắt đầu huấn luyện quá trình Alignment (PPG -> ECG)...")
    for epoch in range(1, CFG.epochs + 1):
        
        train_metrics = train_one_epoch(model, criterion, optimizer, scaler, train_loader, epoch)
        val_metrics = validate_one_epoch(model, criterion, val_loader)

        scheduler.step(val_metrics["total"])

        # IN RA CHI TIẾT CÁC LOSS TẠI EPOCH SUMMARY
        print(f"\n=> Epoch {epoch}/{CFG.epochs} Summary:")
        print(f"   [Train] Total: {train_metrics['total']:.4f} | CORAL: {train_metrics['coral']:.4f} | KL: {train_metrics['kl']:.4f} | Pair(L1): {train_metrics['pair']:.4f}")
        print(f"   [Val]   Total: {val_metrics['total']:.4f} | CORAL: {val_metrics['coral']:.4f} | KL: {val_metrics['kl']:.4f} | Pair(L1): {val_metrics['pair']:.4f}")

        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            bad_epochs = 0
            torch.save(model.state_dict(), save_path)
            print(f"   [+] Đã lưu mô hình Alignment tốt nhất mới!")
        else:
            bad_epochs += 1
            print(f"   [-] Loss tổng không cải thiện trong {bad_epochs} epoch(s).")

        if bad_epochs >= CFG.patience:
            print(f"\n[!] Early Stopping được kích hoạt tại epoch {epoch}.")
            break

if __name__ == "__main__":
    main()