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
import neurokit2 as nk

# Nhớ import đúng tên class model từ file của bạn
from CLIP import PPGtoECGDualBranchReconstructionNet
from load_data import LoadData

# ==========================================
# CONFIG & HYPERPARAMETERS
# ==========================================
CONFIG = {
    "batch_size": 64,
    "epochs": 200,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "sampling_rate": 125,
    "input_len": 2400,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "num_workers": 4,
    "use_amp": True,
    "grad_clip": 1.0,
    "save_every": 10,
    "seed": 42,

    # QRS mask
    "qrs_dilate": 21,

    # Model dims
    "dims": (48, 96, 128, 256),
    "num_blocks_per_stage": 2,
    "num_heads": 8,
    "attn_depth_qrs": 2,
    "attn_depth_non_qrs": 2,
    "drop_path": 0.05,

    # Loss weights
    "lambda_qrs": 1.0,
    "lambda_non": 0.7,
    "lambda_final": 1.2,
}

# ==========================================
# UTILITIES & LOSS FUNCTIONS
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_qrs_mask_neurokit(
    ecg_batch: torch.Tensor,
    sampling_rate: int = 125,
    dilate_size: int = 21,
) -> torch.Tensor:
    """Tạo binary mask cho vùng QRS bằng NeuroKit2."""
    assert ecg_batch.ndim == 3 and ecg_batch.shape[1] == 1, \
        f"Expected ecg_batch [B,1,L], got {ecg_batch.shape}"

    batch_size, _, length = ecg_batch.shape
    mask = torch.zeros_like(ecg_batch, dtype=torch.float32)

    ecg_np = ecg_batch.squeeze(1).detach().cpu().numpy()
    pad = dilate_size // 2

    for i in range(batch_size):
        signal = ecg_np[i]
        try:
            _, info = nk.ecg_peaks(
                signal,
                sampling_rate=sampling_rate,
                method="pantompkins1985",
            )
            rpeaks = info.get("ECG_R_Peaks", [])
            for r in rpeaks:
                start = max(0, int(r) - pad)
                end = min(length, int(r) + pad + 1)
                mask[i, 0, start:end] = 1.0
        except Exception:
            continue
    return mask

def align_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target) + 0.5 * F.mse_loss(pred, target)

def masked_align_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Tính L1 + 0.5*MSE chỉ trên vùng có mask"""
    diff = pred - target
    abs_term = torch.abs(diff) * mask
    sq_term = (diff ** 2) * mask

    l1 = abs_term.sum() / (mask.sum() + eps)
    mse = sq_term.sum() / (mask.sum() + eps)

    return l1 + 0.5 * mse

def pearson_corr_loss(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    """Dùng cho Evaluation (không tham gia backprop)"""
    x = x.squeeze(1)
    y = y.squeeze(1)
    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)
    num = (x * y).sum(dim=1)
    den = torch.sqrt((x.pow(2).sum(dim=1) + eps) * (y.pow(2).sum(dim=1) + eps))
    corr = num / den
    return corr.mean().item()

def plot_losses_combined(history, title='Training Progress'):
    plt.figure(figsize=(12, 8))
    for label, values in history.items():
        if values:
            plt.plot(range(1, len(values) + 1), values, label=label)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True)
    plt.savefig("training_history.png")
    plt.close()

# ==========================================
# CORE TRAINING FUNCTIONS
# ==========================================
def train_epoch(model, dataloader, optimizer, device, scaler, current_epoch):
    model.train()
    metrics = {k: 0.0 for k in ['total', 'final', 'qrs', 'non_qrs']}
    progress_bar = tqdm(dataloader, desc=f"Epoch {current_epoch} [Train]", unit="batch")
    
    for ecg, ppg, _ in progress_bar:
        ecg_target = ecg.to(device).float().unsqueeze(1)
        ppg_input = ppg.to(device).float().unsqueeze(1)
        
        # Tạo mask trên CPU rồi đẩy lên GPU
        qrs_mask = generate_qrs_mask_neurokit(
            ecg_target, 
            sampling_rate=CONFIG["sampling_rate"], 
            dilate_size=CONFIG["qrs_dilate"]
        ).to(device)
        non_qrs_mask = 1.0 - qrs_mask
        
        optimizer.zero_grad()
        
        with autocast(enabled=CONFIG["use_amp"]):
            outputs = model(ppg_raw=ppg_input)
            
            # Tính toán các thành phần Loss
            loss_final = align_loss(outputs["ecg_final_pred"], ecg_target)
            loss_qrs = masked_align_loss(outputs["ecg_qrs_pred"], ecg_target, qrs_mask)
            loss_non_qrs = masked_align_loss(outputs["ecg_non_qrs_pred"], ecg_target, non_qrs_mask)
            
            total_loss = (
                CONFIG["lambda_final"] * loss_final +
                CONFIG["lambda_qrs"] * loss_qrs +
                CONFIG["lambda_non"] * loss_non_qrs
            )

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        metrics['total'] += total_loss.item()
        metrics['final'] += loss_final.item()
        metrics['qrs'] += loss_qrs.item()
        metrics['non_qrs'] += loss_non_qrs.item()

        progress_bar.set_postfix({
            'Total': f"{total_loss.item():.4f}", 
            'Final': f"{loss_final.item():.4f}",
            'QRS': f"{loss_qrs.item():.4f}"
        })

    num_batches = len(dataloader)
    return {k: v / num_batches for k, v in metrics.items()}


def evaluate(model, dataloader, device):
    model.eval()
    t_total, t_final, t_corr = 0.0, 0.0, 0.0
    
    with torch.no_grad():
        for ecg, ppg, _ in dataloader:
            ecg_target = ecg.to(device).float().unsqueeze(1)
            ppg_input = ppg.to(device).float().unsqueeze(1)
            
            qrs_mask = generate_qrs_mask_neurokit(
                ecg_target, sampling_rate=CONFIG["sampling_rate"], dilate_size=CONFIG["qrs_dilate"]
            ).to(device)
            non_qrs_mask = 1.0 - qrs_mask
            
            outputs = model(ppg_raw=ppg_input)
            
            loss_final = align_loss(outputs["ecg_final_pred"], ecg_target)
            loss_qrs = masked_align_loss(outputs["ecg_qrs_pred"], ecg_target, qrs_mask)
            loss_non_qrs = masked_align_loss(outputs["ecg_non_qrs_pred"], ecg_target, non_qrs_mask)
            
            total_loss = (
                CONFIG["lambda_final"] * loss_final +
                CONFIG["lambda_qrs"] * loss_qrs +
                CONFIG["lambda_non"] * loss_non_qrs
            )
            
            t_total += total_loss.item()
            t_final += loss_final.item()
            t_corr += pearson_corr_loss(outputs["ecg_final_pred"], ecg_target)
            
    n = len(dataloader)
    return t_total / n, t_final / n, t_corr / n

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    set_seed(CONFIG["seed"])
    
    # Khởi tạo Dataloaders
    train_loader = DataLoader(LoadData('/home/linhhima/Diffusion datasets/train.npz'), 
                              batch_size=CONFIG["batch_size"], shuffle=True, 
                              num_workers=CONFIG["num_workers"], pin_memory=True)
    test_loader = DataLoader(LoadData('/home/linhhima/Diffusion datasets/val.npz'), 
                             batch_size=CONFIG["batch_size"], shuffle=False, 
                             num_workers=CONFIG["num_workers"], pin_memory=True)
    
    # Khởi tạo Mô hình
    model = PPGtoECGDualBranchReconstructionNet(
        input_len=CONFIG["input_len"],
        dims=CONFIG["dims"],
        num_blocks_per_stage=CONFIG["num_blocks_per_stage"],
        num_heads=CONFIG["num_heads"],
        attn_depth_qrs=CONFIG["attn_depth_qrs"],
        attn_depth_non_qrs=CONFIG["attn_depth_non_qrs"],
        drop_path=CONFIG["drop_path"]
    ).to(CONFIG["device"])

    optimizer = AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scaler = GradScaler(enabled=CONFIG["use_amp"])
    
    history = {'Train Total Loss': [], 'Train Final Loss': [], 'Test Final Loss': [], 'Test Pearson': []}
    best_test_final = float('inf')

    for epoch in range(1, CONFIG["epochs"] + 1):
        start_time = time.time()

        # Huấn luyện
        avg_train = train_epoch(model, train_loader, optimizer, CONFIG["device"], scaler, epoch)
        
        # Đánh giá
        test_total, test_final, test_corr = evaluate(model, test_loader, CONFIG["device"])
        
        # Ghi nhận lịch sử
        history['Train Total Loss'].append(avg_train['total'])
        history['Train Final Loss'].append(avg_train['final'])
        history['Test Final Loss'].append(test_final)
        history['Test Pearson'].append(test_corr)

        print(f"Epoch {epoch} Summary (Time: {time.time() - start_time:.2f}s):")
        print(f"[Train] Total: {avg_train['total']:.4f} | Final: {avg_train['final']:.4f} | QRS: {avg_train['qrs']:.4f} | Non-QRS: {avg_train['non_qrs']:.4f}")
        print(f"[Test]  Total: {test_total:.4f} | Final (L1+MSE): {test_final:.4f} | Pearson Corr: {test_corr:.4f}")

        # Lưu checkpoint
        if test_final < best_test_final:
            best_test_final = test_final
            torch.save(model.state_dict(), "best_dual_branch_model.pth")
            print(f"*** New Best Model Saved (Final Loss: {test_final:.4f} at Epoch {epoch})")
            
        # if epoch % CONFIG["save_every"] == 0:
        #     torch.save(model.state_dict(), f"checkpoint_epoch_{epoch}.pth")

    print("Training Finished. Generating final plots...")
    # plot_losses_combined(history, title='Dual-Branch Reconstruction Training')