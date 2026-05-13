import os
import random
import numpy as np
from dataclasses import dataclass, asdict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from load_data import LoadData
from ecg2ecg import ECGAutoencoder, ECGAEConfig, ECGReconstructionLoss

# ============================================================
# 1. Cấu hình (Config) - Đã rút gọn
# ============================================================
@dataclass
class TrainConfig:
    seed: int = 42

    train_path: str = "/home/linhhima/Diffusion datasets/train.npz"
    val_path: str = "/home/linhhima/Diffusion datasets/val.npz"

    # Chỉ dùng 1 file lưu model duy nhất
    save_dir: str = "saved_models_ecg_vae"
    best_model_name: str = "best_ecg_autoencoder.pth"

    batch_size: int = 128
    epochs: int = 100
    num_workers: int = 4

    lr: float = 1e-4
    weight_decay: float = 1e-4

    grad_clip: float = 1.0
    use_amp: bool = True
    patience: int = 20  # Ngừng sớm nếu sau 20 epoch loss không giảm

    # Thông số Model
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


CFG = TrainConfig()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(CFG.save_dir, exist_ok=True)

# ============================================================
# 2. Tiện ích (Utils)
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

# ============================================================
# 3. Hàm Huấn luyện & Đánh giá (Train / Eval)
# ============================================================
def train_one_epoch(model, criterion, optimizer, scaler, dataloader, epoch):
    model.train()
    total_loss_sum = 0.0

    loop = tqdm(dataloader, desc=f"Train Epoch {epoch}/{CFG.epochs}")
    for ecg, _, _ in loop:
        ecg = ecg.float().unsqueeze(1).to(DEVICE) if ecg.dim() == 2 else ecg.float().to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=CFG.use_amp):
            outputs = model(ecg)
            losses = criterion(outputs, ecg)
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

    for ecg, _, _ in dataloader:
        ecg = ecg.float().unsqueeze(1).to(DEVICE) if ecg.dim() == 2 else ecg.float().to(DEVICE)

        outputs = model(ecg)
        losses = criterion(outputs, ecg)
        
        total_loss_sum += losses["loss_total"].item()

    return total_loss_sum / len(dataloader)

# ============================================================
# 4. Vòng lặp Chính (Main)
# ============================================================
def main():
    set_seed(CFG.seed)
    print(f"[*] Đang sử dụng thiết bị: {DEVICE}")

    train_loader, val_loader = get_dataloaders()

    model_cfg = ECGAEConfig(
        input_length=CFG.input_length, in_channels=CFG.in_channels,
        dims=CFG.dims, depths=CFG.depths, latent_channels=CFG.latent_channels,
        latent_length=CFG.latent_length, attn_heads=CFG.attn_heads,
        attn_dropout=CFG.attn_dropout, global_latent_dim=CFG.global_latent_dim,
        trend_poly=CFG.trend_poly,
    )

    model = ECGAutoencoder(model_cfg).to(DEVICE)
    
    # Khởi tạo Loss tối giản (Đã loại bỏ recon_mode)
    criterion = ECGReconstructionLoss().to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    
    # Giảm learning rate nếu validation loss không cải thiện
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=6, verbose=True
    )

    scaler = GradScaler(enabled=CFG.use_amp)

    best_val_loss = float("inf")
    bad_epochs = 0
    save_path = os.path.join(CFG.save_dir, CFG.best_model_name)

    print("\n[*] Bắt đầu quá trình huấn luyện...")
    for epoch in range(1, CFG.epochs + 1):
        # 1. Chạy Huấn luyện
        train_loss = train_one_epoch(model, criterion, optimizer, scaler, train_loader, epoch)
        
        # 2. Chạy Đánh giá
        val_loss = validate_one_epoch(model, criterion, val_loader)

        # 3. Cập nhật Learning Rate
        scheduler.step(val_loss)

        print(f" -> Kết quả: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}")

        # 4. Kiểm tra lưu model & Early Stopping dựa trên 1 Loss duy nhất
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            bad_epochs = 0
            save_checkpoint(save_path, epoch, model, optimizer, best_val_loss, model_cfg, CFG)
            print(f"   [+] Đã lưu mô hình tốt nhất mới tại epoch {epoch}!")
        else:
            bad_epochs += 1
            print(f"   [-] Loss không cải thiện trong {bad_epochs} epoch liên tiếp.")

        if bad_epochs >= CFG.patience:
            print(f"\n[!] Kích hoạt Early Stopping: Dừng sớm tại epoch {epoch} do loss không giảm.")
            break

    print("\n[*] Huấn luyện hoàn tất!")

if __name__ == "__main__":
    main()