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

# Import các hàm và Model của Stage 1
from load_data import LoadData
from ecg2ecg import ECGAutoencoder, ECGAEConfig
from ppg2ecg import PPG2ECGModel, PPG2ECGConfig
from flow_model import LatentRectifiedFlow

# ============================================================
# 1. CẤU HÌNH CHO GIAI ĐOẠN 2 (STAGE 2 CONFIG)
# ============================================================
@dataclass
class Stage2TrainConfig:
    seed: int = 42
    
    # Data paths
    train_path: str = "/home/linhhima/Diffusion datasets/train.npz"
    val_path: str = "/home/linhhima/Diffusion datasets/val.npz"

    # Trọng số của Stage 1 (Phải chạy xong Giai đoạn 1 mới có)
    ecg_checkpoint_path: str = "/home/linhhima/PPG_ECG/Result_model2/saved_models_ecg_vae/best_ecg_autoencoder.pth" 
    ppg_checkpoint_path: str = "/home/linhhima/PPG_ECG/Result_model2/saved_models_alignment_batch_32/best_ppg_alignment.pth"
    
    # Đường dẫn lưu mô hình Stage 2
    save_dir: str = "saved_models_flow_cfg" # Đổi tên thư mục một chút để phân biệt
    best_model_name: str = "best_rectified_flow_cfg.pth"

    # Training Params cho Rectified Flow
    batch_size: int = 128  
    epochs: int = 300
    num_workers: int = 4
    lr: float = 2e-4       
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    use_amp: bool = True
    patience: int = 30
    
    # MỚI: Tỷ lệ drop condition cho Classifier-Free Guidance
    cfg_drop_prob: float = 0.1 

    # Kiến trúc không gian Latent (Kế thừa từ Stage 1)
    input_length: int = 2400
    latent_channels: int = 16
    latent_length: int = 75
    dims: tuple = (64, 128, 256, 512)
    depths: tuple = (2, 2, 4, 2)
    
    # Kiến trúc Flow Model
    flow_hidden_dim: int = 128
    flow_num_blocks: int = 6

CFG = Stage2TrainConfig()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(CFG.save_dir, exist_ok=True)

# ============================================================
# 3. UTILS & DATA LOADER
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
# 4. LOAD FROZEN STAGE 1 MODELS
# ============================================================
def load_frozen_stage1():
    print("[*] Đang tải và đóng băng các mô hình Stage 1 (ECG Teacher & PPG Encoder)...")
    
    ecg_cfg = ECGAEConfig(
        input_length=CFG.input_length, in_channels=1, dims=CFG.dims, depths=CFG.depths,
        latent_channels=CFG.latent_channels, latent_length=CFG.latent_length,
        attn_heads=8, attn_dropout=0.0, global_latent_dim=128, trend_poly=2
    )
    ppg_cfg = PPG2ECGConfig(
        input_length=CFG.input_length, ppg_in_channels=1, dims=CFG.dims, depths=CFG.depths,
        latent_channels=CFG.latent_channels, latent_length=CFG.latent_length,
        attn_heads=8, attn_dropout=0.0, use_derivatives=True, proj_dim=128
    )

    ecg_ae = ECGAutoencoder(ecg_cfg).to(DEVICE)
    if not os.path.exists(CFG.ecg_checkpoint_path):
        raise FileNotFoundError(f"Lỗi: Không tìm thấy trọng số ECG Teacher tại {CFG.ecg_checkpoint_path}")
    ckpt_ecg = torch.load(CFG.ecg_checkpoint_path, map_location=DEVICE)
    ecg_ae.load_state_dict(ckpt_ecg.get("model_state_dict", ckpt_ecg))
    ecg_ae.eval()
    for p in ecg_ae.parameters(): p.requires_grad = False

    ppg_model = PPG2ECGModel(ecg_ae=ecg_ae, cfg=ppg_cfg).to(DEVICE)
    if not os.path.exists(CFG.ppg_checkpoint_path):
        raise FileNotFoundError(f"Lỗi: Không tìm thấy trọng số PPG Alignment tại {CFG.ppg_checkpoint_path}")
    ckpt_ppg = torch.load(CFG.ppg_checkpoint_path, map_location=DEVICE)
    ppg_model.load_state_dict(ckpt_ppg.get("model_state_dict", ckpt_ppg))
    ppg_model.eval()
    for p in ppg_model.parameters(): p.requires_grad = False

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

        # 1. TRÍCH XUẤT LATENT TỪ MÔ HÌNH ĐÓNG BĂNG
        with torch.no_grad():
            feat_ecg = ecg_ae.encoder(ecg)
            z_ecg = ecg_ae.latent_head(feat_ecg)   # Target
           
            feat_ppg = ppg_model.ppg_encoder(ppg)
            z_ppg = ppg_model.ppg_latent_head(feat_ppg) # Condition

        # MỚI: Logic Classifier-Free Guidance (CFG) Drop
        # Tạo một mask quyết định xem sample nào trong batch sẽ bị mất điều kiện (unconditional)
        # Xác suất drop = 0.1 (10% sẽ không có điều kiện PPG)
        drop_mask = torch.rand((B, 1, 1), device=DEVICE) < CFG.cfg_drop_prob
        # Những chỗ drop_mask = True, z_ppg sẽ bị ép về 0
        z_ppg_cond = torch.where(drop_mask, torch.zeros_like(z_ppg), z_ppg)

        # 2. THIẾT LẬP RECTIFIED FLOW
        # z0 = torch.randn_like(z_ecg) # Lấy mẫu nhiễu N(0, I)
        z0 = z_ppg
        
        t = torch.rand((B,), device=DEVICE)
        t_expand = t.view(B, 1, 1) 
        
        xt = t_expand * z_ecg + (1.0 - t_expand) * z0
        v_target = z_ecg - z0

        # 3. HUẤN LUYỆN
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=CFG.use_amp):
            # Dự đoán trường vector v với điều kiện z_ppg_cond (đã bị drop ngẫu nhiên)
            v_pred = flow_model(xt, t, z_ppg_cond)
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

        # z0 = torch.randn_like(z_ecg)
        z0 = z_ppg
        t = torch.rand((B,), device=DEVICE)
        t_expand = t.view(B, 1, 1)
        
        xt = t_expand * z_ecg + (1.0 - t_expand) * z0
        v_target = z_ecg - z0

        # Trong Validation, ta đo Loss với đầy đủ điều kiện (Không Drop)
        v_pred = flow_model(xt, t, z_ppg)
        loss = F.mse_loss(v_pred, v_target)

        total_loss += loss.item()

    return total_loss / len(dataloader)

# ============================================================
# 6. HÀM MAIN
# ============================================================
def main():
    set_seed(CFG.seed)
    print(f"[*] Đang sử dụng thiết bị: {DEVICE}")

    train_loader, val_loader = get_dataloaders()
    ecg_ae, ppg_model = load_frozen_stage1()

    flow_model = LatentRectifiedFlow(
        latent_channels=CFG.latent_channels, 
        cond_channels=CFG.latent_channels, 
        hidden_dim=CFG.flow_hidden_dim, 
        num_blocks=CFG.flow_num_blocks
    ).to(DEVICE)

    optimizer = optim.AdamW(flow_model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, verbose=True)
    scaler = GradScaler(enabled=CFG.use_amp)

    best_val_loss = float("inf")
    bad_epochs = 0
    save_path = os.path.join(CFG.save_dir, CFG.best_model_name)

    print("\n" + "="*50)
    print("[*] BẮT ĐẦU HUẤN LUYỆN GIAI ĐOẠN 2: LATENT RECTIFIED FLOW VỚI CFG")
    print("="*50 + "\n")

    for epoch in range(1, CFG.epochs + 1):
        
        train_mse = train_one_epoch(flow_model, ecg_ae, ppg_model, optimizer, scaler, train_loader, epoch)
        val_mse = val_one_epoch(flow_model, ecg_ae, ppg_model, val_loader)

        scheduler.step(val_mse)

        print(f"   => [Epoch {epoch}] Train MSE: {train_mse:.5f} | Val MSE: {val_mse:.5f}")

        if val_mse < best_val_loss:
            best_val_loss = val_mse
            bad_epochs = 0
            torch.save(flow_model.state_dict(), save_path)
            print(f"   [+] Đã lưu mô hình Flow tốt nhất (Val MSE giảm xuống {best_val_loss:.5f})!")
        else:
            bad_epochs += 1
            print(f"   [-] Loss Val không cải thiện trong {bad_epochs} epoch(s).")

        if bad_epochs >= CFG.patience:
            print(f"\n[!] Kích hoạt Early Stopping tại epoch {epoch}.")
            break

if __name__ == "__main__":
    main()