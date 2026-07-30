import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

from load_data import LoadData
from ecg2ecg import ECGAutoencoder, ECGAEConfig
from ppg2ecg import PPG2ECGModel, PPG2ECGConfig, CardioAlignLoss

@dataclass
class TrainConfig:
    seed: int = 42
    
    train_path: str = "/home/linhhima/Diffusion_datasets/combined_segment_split_train.npz"
    val_path: str = "/home/linhhima/Diffusion_datasets/combined_segment_split_val.npz"

    ecg_checkpoint_path: str = "/home/linhhima/PPG_ECG/PPG2ECG_RF/Proposed/saved_models_ecg_vae_segment/best_ecg_autoencoder.pth" 
    ppg_pretrained_path: str = "/home/linhhima/PPG_ECG/PPG2ECG_RF/Pretrained_no_mse/saved_models_ppg_vae_segment/best_ppg_autoencoder.pth"

    save_dir: str = "/home/linhhima/PPG_ECG/PPG2ECG_RF/Pretrained_no_mse/saved_models_alignment_subject"
    best_model_name: str = "best_ppg_alignment.pth"
    plot_name: str = "alignment_loss_curves.png"

    batch_size: int = 128
    epochs: int = 200
    num_workers: int = 4
    lr: float = 5e-5  
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    use_amp: bool = True
    patience: int = 20

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

    use_derivatives: bool = True

    coral_weight: float = 1.0
    kl_weight: float = 1.0

CFG = TrainConfig()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(CFG.save_dir, exist_ok=True)

def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
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

def plot_alignment_losses(history, save_dir, filename):
    epochs = range(1, len(history["train_total"]) + 1)
    metrics = ["total", "coral", "kl"]
    titles = ["Total Loss", "CORAL Loss", "KL Divergence Loss"]
    colors = {"train": "blue", "val": "red"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        axes[i].plot(epochs, history[f"train_{metric}"], label="Train", color=colors["train"], linewidth=1.5)
        axes[i].plot(epochs, history[f"val_{metric}"], label="Val", color=colors["val"], linewidth=1.5, linestyle="--")
        
        axes[i].set_title(titles[i], fontsize=12, fontweight="bold")
        axes[i].set_xlabel("Epochs", fontsize=10)
        axes[i].set_ylabel("Loss Value", fontsize=10)
        axes[i].grid(True, linestyle=":", alpha=0.6)
        axes[i].legend(fontsize=10)

    plt.suptitle("PPG-ECG Alignment Finetuning Metrics", fontsize=16, fontweight="bold")
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, filename)
    plt.savefig(plot_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"\n[+] Finetuning Loss chart plotted and saved at: {plot_path}")

def build_frozen_ecg_teacher():
    print(f"[*] Loading ECG Teacher model from: {CFG.ecg_checkpoint_path}")
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
    
    torch.serialization.add_safe_globals([ECGAEConfig])
    checkpoint = torch.load(CFG.ecg_checkpoint_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        ecg_ae.load_state_dict(checkpoint["model_state_dict"])
    else:
        ecg_ae.load_state_dict(checkpoint)

    ecg_ae.eval()
    for p in ecg_ae.parameters():
        p.requires_grad = False
        
    return ecg_ae

def load_pretrained_ppg_weights(model: nn.Module):
    print(f"[*] Loading Pre-trained PPG weights from: {CFG.ppg_pretrained_path}")
    if not os.path.exists(CFG.ppg_pretrained_path):
        print(f"[!] Warning: Pre-trained PPG checkpoint not found at {CFG.ppg_pretrained_path}. Training from scratch!")
        return model

    from ppg2ppg import PPGAEConfig
    torch.serialization.add_safe_globals([PPGAEConfig])
    
    pretrained_dict = torch.load(CFG.ppg_pretrained_path, map_location=DEVICE)
    if isinstance(pretrained_dict, dict) and "model_state_dict" in pretrained_dict:
        state_dict = pretrained_dict["model_state_dict"]
    else:
        state_dict = pretrained_dict

    model_dict = model.state_dict()
    mapped_state_dict = {}

    for k, v in state_dict.items():
        if k.startswith("encoder."):
            new_k = k.replace("encoder.", "ppg_encoder.")
            if new_k in model_dict:
                mapped_state_dict[new_k] = v
        elif k.startswith("latent_head."):
            new_k = k.replace("latent_head.", "ppg_latent_head.")
            if new_k in model_dict:
                mapped_state_dict[new_k] = v

    print(f"[+] Successfully mapped {len(mapped_state_dict)} tensors for Finetuning.")
    model_dict.update(mapped_state_dict)
    model.load_state_dict(model_dict)
    return model

def train_one_epoch(model, criterion, optimizer, scaler, dataloader, epoch):
    model.train()
    running = {"total": 0.0, "coral": 0.0, "kl": 0.0}

    loop = tqdm(dataloader, desc=f"Train Finetune Epoch {epoch}/{CFG.epochs}")
    for ecg, ppg, _ in loop:
        ecg = ecg.float().unsqueeze(1).to(DEVICE) if ecg.dim() == 2 else ecg.float().to(DEVICE)
        ppg = ppg.float().unsqueeze(1).to(DEVICE) if ppg.dim() == 2 else ppg.float().to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=CFG.use_amp):
            outputs = model(ppg=ppg, ecg=ecg)
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

        loop.set_postfix(
            Tot=f"{loss_total.item():.3f}",
            Coral=f"{align_losses['loss_coral'].item():.4f}",
            KL=f"{align_losses['loss_kl'].item():.3f}"
        )

    n = len(dataloader)
    return {k: v / n for k, v in running.items()}

@torch.no_grad()
def validate_one_epoch(model, criterion, dataloader):
    model.eval()
    running = {"total": 0.0, "coral": 0.0, "kl": 0.0}

    for ecg, ppg, _ in dataloader:
        ecg = ecg.float().unsqueeze(1).to(DEVICE) if ecg.dim() == 2 else ecg.float().to(DEVICE)
        ppg = ppg.float().unsqueeze(1).to(DEVICE) if ppg.dim() == 2 else ppg.float().to(DEVICE)

        outputs = model(ppg=ppg, ecg=ecg)
        align_losses = criterion(outputs)
        loss_total = align_losses["loss_total"]

        running["total"] += loss_total.item()
        running["coral"] += align_losses["loss_coral"].item()
        running["kl"] += align_losses["loss_kl"].item()

    n = len(dataloader)
    return {k: v / n for k, v in running.items()}

def main():
    set_seed(CFG.seed)
    print(f"[*] Using device: {DEVICE}")

    train_loader, val_loader = get_dataloaders()

    ecg_teacher = build_frozen_ecg_teacher()

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
    )

    model = PPG2ECGModel(ecg_ae=ecg_teacher, cfg=ppg_cfg).to(DEVICE)

    model = load_pretrained_ppg_weights(model)

    criterion = CardioAlignLoss(
        coral_weight=CFG.coral_weight, 
        kl_weight=CFG.kl_weight
    ).to(DEVICE)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    scaler = GradScaler(enabled=CFG.use_amp)

    best_val_loss = float("inf")
    bad_epochs = 0
    save_path = os.path.join(CFG.save_dir, CFG.best_model_name)

    history = {
        "train_total": [], "train_coral": [], "train_kl": [],
        "val_total": [], "val_coral": [], "val_kl": []
    }

    print("\n[*] Starting Alignment Finetuning process (PPG -> ECG)...")
    current_lr = optimizer.param_groups[0]['lr']

    for epoch in range(1, CFG.epochs + 1):
        
        train_metrics = train_one_epoch(model, criterion, optimizer, scaler, train_loader, epoch)
        val_metrics = validate_one_epoch(model, criterion, val_loader)

        for metric in ["total", "coral", "kl"]:
            history[f"train_{metric}"].append(train_metrics[metric])
            history[f"val_{metric}"].append(val_metrics[metric])

        scheduler.step(val_metrics["total"])
        
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != current_lr:
            print(f"[!] Learning Rate decreased from {current_lr} to {new_lr}")
            current_lr = new_lr

        print(f"\n=> Epoch {epoch}/{CFG.epochs} Summary:")
        print(f"   [Train] Total: {train_metrics['total']:.4f} | CORAL: {train_metrics['coral']:.4f} | KL: {train_metrics['kl']:.4f}")
        print(f"   [Val]   Total: {val_metrics['total']:.4f} | CORAL: {val_metrics['coral']:.4f} | KL: {val_metrics['kl']:.4f}")

        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            bad_epochs = 0
            torch.save(model.state_dict(), save_path)
            print(f"   [+] New best Alignment finetuned model saved!")
        else:
            bad_epochs += 1
            print(f"   [-] Total loss did not improve for {bad_epochs} epoch(s).")

        if bad_epochs >= CFG.patience:
            print(f"\n[!] Early Stopping triggered at epoch {epoch}.")
            break

    print("\n[*] Finetuning complete!")
    
    plot_alignment_losses(history, CFG.save_dir, CFG.plot_name)

if __name__ == "__main__":
    main()