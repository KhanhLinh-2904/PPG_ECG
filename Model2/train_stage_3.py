import os
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

from load_data import LoadData 
from ecg2ecg import ECGAutoencoder, ECGAEConfig
from ppg2ecg import PPG2ECGModel, PPG2ECGConfig 
from flow_model import LatentRectifiedFlow

# ============================================================
# 1. CẤU HÌNH CHO GIAI ĐOẠN 3 (STAGE 3 CONFIG)
# ============================================================
@dataclass
class Stage3TrainConfig:
    seed: int = 42
    
    # Data paths
    train_path: str = "/home/linhhima/Diffusion datasets/train.npz"
    val_path: str = "/home/linhhima/Diffusion datasets/val.npz"

    # Trọng số của Stage 1 & Stage 2 (Phải chạy xong mới có)
    ecg_checkpoint_path: str = "/home/linhhima/PPG_ECG/saved_models_ecg_vae/best_ecg_autoencoder.pth" 
    ppg_checkpoint_path: str = "/home/linhhima/PPG_ECG/saved_models_alignment_batch_32/best_ppg_alignment.pth"
    flow_checkpoint_path: str = "/home/linhhima/PPG_ECG/saved_models_flow_1024/best_rectified_flow.pth"
    
    # Đường dẫn lưu mô hình Decoder đã Finetune
    save_dir: str = "saved_models_finetune"
    best_model_name: str = "best_finetuned_decoder.pth"

    # Training Params
    batch_size: int = 128
    epochs: int = 100
    num_workers: int = 4
    lr: float = 5e-5       # Tốc độ học nhỏ hơn vì đây là Finetuning
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    use_amp: bool = True
    patience: int = 15

    # Cấu hình Latent & Model
    input_length: int = 2400
    latent_channels: int = 16
    ODE_STEPS: int = 10     # Lấy số bước nhỏ để train nhanh hơn, vì chỉ cần sinh z_ecg_hat

CFG = Stage3TrainConfig()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(CFG.save_dir, exist_ok=True)

# ============================================================
# 2. UTILS & DATA LOADER
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
# 3. HÀM GIẢI ODE (Euler Solver)
# ============================================================
# Ở giai đoạn train, ta KHÔNG dùng torch.no_grad() trong euler_solve 
# vì cần gradient truyền qua z_ecg_hat về lại Flow (nếu muốn) 
# Tuy nhiên, trong Stage 3 này ta đóng băng Flow, nên vẫn có thể dùng no_grad 
# cho phần sinh z_ecg_hat để tiết kiệm VRAM.

@torch.no_grad()
def generate_z_ecg_hat(flow_model, z_ppg, num_steps=10):
    B, C, L = z_ppg.shape
    xt = torch.randn((B, C, L), device=DEVICE)
    dt = 1.0 / num_steps
    
    for step in range(num_steps):
        t_val = step * dt
        t_tensor = torch.full((B,), t_val, device=DEVICE)
        v_pred = flow_model(xt, t_tensor, z_ppg)
        xt = xt + v_pred * dt
        
    return xt

# ============================================================
# 4. TẢI VÀ ĐÓNG BĂNG/MỞ KHÓA MÔ HÌNH
# ============================================================
def setup_models_for_stage3():
    print(f"[*] Đang tải các mô hình trên {DEVICE}...")
    
    # 1. Load ECG Autoencoder
    ecg_cfg = ECGAEConfig(
        input_length=CFG.input_length, in_channels=1, dims=(64, 128, 256, 512),
        depths=(2, 2, 4, 2), latent_channels=CFG.latent_channels, latent_length=75,
        attn_heads=8, attn_dropout=0.0, global_latent_dim=128, trend_poly=2
    )
    ecg_ae = ECGAutoencoder(ecg_cfg).to(DEVICE)
    ckpt_ecg = torch.load(CFG.ecg_checkpoint_path, map_location=DEVICE)
    ecg_ae.load_state_dict(ckpt_ecg.get("model_state_dict", ckpt_ecg))
    
    # Đóng băng Encoder của ECG AE
    for p in ecg_ae.encoder.parameters(): p.requires_grad = False
    for p in ecg_ae.latent_head.parameters(): p.requires_grad = False
    # MỞ KHÓA (UNFREEZE) DECODER
    for p in ecg_ae.decoder.parameters(): p.requires_grad = True

    # 2. Load PPG2ECG Model
    ppg_cfg = PPG2ECGConfig(
        input_length=CFG.input_length, ppg_in_channels=1, dims=(64, 128, 256, 512),
        depths=(2, 2, 4, 2), latent_channels=CFG.latent_channels, latent_length=75,
        attn_heads=8, attn_dropout=0.0, use_derivatives=True, proj_dim=128
    )
    ppg_model = PPG2ECGModel(ecg_ae=ecg_ae, cfg=ppg_cfg).to(DEVICE)
    ckpt_ppg = torch.load(CFG.ppg_checkpoint_path, map_location=DEVICE)
    ppg_model.load_state_dict(ckpt_ppg.get("model_state_dict", ckpt_ppg))
    
    # Đóng băng toàn bộ PPG Model
    ppg_model.eval()
    for p in ppg_model.parameters(): p.requires_grad = False

    # 3. Load Flow Model
    flow_model = LatentRectifiedFlow(latent_channels=16, cond_channels=16, hidden_dim=128, num_blocks=6).to(DEVICE)
    flow_model.load_state_dict(torch.load(CFG.flow_checkpoint_path, map_location=DEVICE))
    
    # Đóng băng toàn bộ Flow Model
    flow_model.eval()
    for p in flow_model.parameters(): p.requires_grad = False
    
    return ecg_ae, ppg_model, flow_model

# ============================================================
# 5. TRAINING LOOPS
# ============================================================
def train_one_epoch(ecg_ae, ppg_model, flow_model, optimizer, scaler, dataloader, epoch):
    # Chỉ bật chế độ train cho Decoder
    ecg_ae.decoder.train()
    total_loss = 0.0

    loop = tqdm(dataloader, desc=f"Train Epoch {epoch}/{CFG.epochs}")
    for ecg, ppg, _ in loop:
        ecg = ecg.float().unsqueeze(1).to(DEVICE) if ecg.dim() == 2 else ecg.float().to(DEVICE)
        ppg = ppg.float().unsqueeze(1).to(DEVICE) if ppg.dim() == 2 else ppg.float().to(DEVICE)

        # Trích xuất latent (Bằng nhánh đã đóng băng)
        with torch.no_grad():
            feat_ppg = ppg_model.ppg_encoder(ppg)
            z_ppg = ppg_model.ppg_latent_head(feat_ppg)
            
            # Giải ODE sinh z_ecg_hat
            z_ecg_hat = generate_z_ecg_hat(flow_model, z_ppg, num_steps=CFG.ODE_STEPS)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=CFG.use_amp):
            # Truyền z_ecg_hat qua Decoder để sinh sóng
            reconstructed_ecg = ecg_ae.decoder(z_ecg_hat)
            
            # Tính Loss giữa Sóng tái tạo và Sóng thật
            # Bạn có thể dùng L1 Loss hoặc MSE Loss
            loss = F.l1_loss(reconstructed_ecg, ecg)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(ecg_ae.decoder.parameters(), CFG.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loop.set_postfix(L1_Loss=f"{loss.item():.4f}")

    return total_loss / len(dataloader)


@torch.no_grad()
def val_one_epoch(ecg_ae, ppg_model, flow_model, dataloader):
    ecg_ae.decoder.eval()
    total_loss = 0.0

    for ecg, ppg, _ in dataloader:
        ecg = ecg.float().unsqueeze(1).to(DEVICE) if ecg.dim() == 2 else ecg.float().to(DEVICE)
        ppg = ppg.float().unsqueeze(1).to(DEVICE) if ppg.dim() == 2 else ppg.float().to(DEVICE)

        feat_ppg = ppg_model.ppg_encoder(ppg)
        z_ppg = ppg_model.ppg_latent_head(feat_ppg)
        z_ecg_hat = generate_z_ecg_hat(flow_model, z_ppg, num_steps=CFG.ODE_STEPS)

        reconstructed_ecg = ecg_ae.decoder(z_ecg_hat)
        loss = F.l1_loss(reconstructed_ecg, ecg)
        
        total_loss += loss.item()

    return total_loss / len(dataloader)

# ============================================================
# 6. MAIN
# ============================================================
def main():
    set_seed(CFG.seed)
    print(f"[*] Đang sử dụng thiết bị: {DEVICE}")

    train_loader, val_loader = get_dataloaders()
    ecg_ae, ppg_model, flow_model = setup_models_for_stage3()

    # Lưu ý: Chỉ truyền các tham số của ecg_ae.decoder vào Optimizer
    optimizer = optim.AdamW(
        ecg_ae.decoder.parameters(), 
        lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)
    scaler = GradScaler(enabled=CFG.use_amp)

    best_val_loss = float("inf")
    bad_epochs = 0
    save_path = os.path.join(CFG.save_dir, CFG.best_model_name)

    print("\n" + "="*50)
    print("[*] BẮT ĐẦU HUẤN LUYỆN GIAI ĐOẠN 3: FINETUNE DECODER")
    print("="*50 + "\n")

    for epoch in range(1, CFG.epochs + 1):
        
        train_loss = train_one_epoch(ecg_ae, ppg_model, flow_model, optimizer, scaler, train_loader, epoch)
        val_loss = val_one_epoch(ecg_ae, ppg_model, flow_model, val_loader)

        scheduler.step(val_loss)

        print(f"   => [Epoch {epoch}] Train L1: {train_loss:.5f} | Val L1: {val_loss:.5f}")

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            bad_epochs = 0
            
            # Chỉ lưu state_dict của riêng Decoder để sau này dễ ghép nối
            torch.save(ecg_ae.decoder.state_dict(), save_path)
            print(f"   [+] Đã lưu Decoder tốt nhất (Val L1 giảm xuống {best_val_loss:.5f})!")
        else:
            bad_epochs += 1
            print(f"   [-] Loss Val không cải thiện trong {bad_epochs} epoch(s).")

        # Early Stopping
        if bad_epochs >= CFG.patience:
            print(f"\n[!] Kích hoạt Early Stopping tại epoch {epoch}.")
            break

if __name__ == "__main__":
    main()