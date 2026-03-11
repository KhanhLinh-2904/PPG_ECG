import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import numpy as np
import random
from load_data import LoadData

from CLIP import PPG2ECG_Model, stockwell_transform_v2
from loss import SoftTargetDistillationLoss, PearsonCorrelationLoss, SpectralLoss

# --- (CONSTANTS) ---
SEED = 44
NUM_EPOCHS = 200
LEARNING_RATE = 1e-4

# 🔥 TỐI ƯU HÓA VRAM: Hạ Batch Size thực tế, tăng số bước cộng dồn
BATCH_SIZE = 4            # Mỗi lần chỉ nhét 4 mẫu vào GPU (Tránh OOM tuyệt đối)
ACCUMULATION_STEPS = 16   # Cộng dồn 16 lần -> Effective Batch Size = 4 * 16 = 64

# Trọng số cân bằng "Tứ Trụ Loss"
WEIGHT_DISTILL = 1.0    # Ép PPG hiểu ECG
WEIGHT_L1 = 1.0         # Ép Mamba vẽ đúng biên độ tuyệt đối
WEIGHT_SPECTRAL = 0.1   # Ép Mamba vẽ đúng nhịp điệu (Thay thế DTW)
WEIGHT_PEARSON = 0.5    # Ép Mamba vẽ đúng hình dáng y khoa

FS = 125 # Tần số lấy mẫu (Hz)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def plot_losses_combined(losses_dict, title='Training Losses'):
    epochs = range(1, len(next(iter(losses_dict.values()))) + 1)
    plt.figure(figsize=(10, 6))
    
    for label, loss_values in losses_dict.items():
        plt.plot(epochs, loss_values, label=label, linewidth=2)
        
    plt.title(title, fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}.png")
    plt.close()

def batch_stockwell_transform(signals_1d_tensor, fs, fmin=0.0, fmax=15.0):
    device = signals_1d_tensor.device
    signals_np = signals_1d_tensor.detach().cpu().numpy()
    batch_st = []
    
    for i in range(signals_np.shape[0]):
        st_matrix = stockwell_transform_v2(signals_np[i], fs, fmin, fmax)
        real_part = np.real(st_matrix)
        imag_part = np.imag(st_matrix)
        stacked = np.stack((real_part, imag_part), axis=0)
        batch_st.append(stacked)
        
    return torch.tensor(np.array(batch_st), dtype=torch.float32).to(device)

def train_epoch_combined(model, dataloader, optimizer, criterion_dict, loss_weights, device, scaler):
    model.train()
    metrics = {'total': 0.0, 'distill': 0.0, 'l1': 0.0, 'spectral': 0.0, 'pearson': 0.0}
    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    # Đưa zero_grad ra khỏi vòng lặp lô nhỏ
    optimizer.zero_grad()

    for i, batch in enumerate(progress_bar):
        ppg_1d, ecg_1d = batch[0].to(device), batch[1].to(device)

        ppg_st_2d = batch_stockwell_transform(ppg_1d, fs=FS, fmin=0.5, fmax=20.0)
        ecg_st_2d = batch_stockwell_transform(ecg_1d, fs=FS, fmin=0.5, fmax=20.0)

        with autocast():
            ppg_embed, ecg_embed, recon_ecg_1d = model(ppg_st_2d, ecg_st_2d)

            loss_distill = criterion_dict['distill'](ppg_embed, ecg_embed)
            loss_l1 = criterion_dict['l1'](recon_ecg_1d, ecg_1d)
            loss_spectral = criterion_dict['spectral'](recon_ecg_1d, ecg_1d)
            loss_pearson = criterion_dict['pearson'](recon_ecg_1d, ecg_1d)

            total_loss = (loss_weights['distill'] * loss_distill +
                          loss_weights['l1'] * loss_l1 +
                          loss_weights['spectral'] * loss_spectral +
                          loss_weights['pearson'] * loss_pearson)
            
            # 🔥 CHIA LOSS: Cực kỳ quan trọng để toán học đúng khi cộng dồn
            loss = total_loss / ACCUMULATION_STEPS

        # Tích lũy gradient (Lưu ý: chưa cập nhật trọng số vội)
        scaler.scale(loss).backward()
        
        # 🔥 CẬP NHẬT TRỌNG SỐ THEO CHU KỲ (Gradient Accumulation)
        if ((i + 1) % ACCUMULATION_STEPS == 0) or ((i + 1) == len(dataloader)):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            # Khởi tạo lại gradient bằng 0 cho đợt cộng dồn tiếp theo
            optimizer.zero_grad()

        # Tính toán metrics để hiển thị (Dùng total_loss nguyên bản, không chia)
        metrics['total'] += total_loss.item()
        metrics['distill'] += loss_distill.item() * loss_weights['distill']
        metrics['l1'] += loss_l1.item() * loss_weights['l1']
        metrics['spectral'] += loss_spectral.item() * loss_weights['spectral']
        metrics['pearson'] += loss_pearson.item() * loss_weights['pearson']

        progress_bar.set_postfix({
            'Tot': f"{total_loss.item():.3f}", 
            'Dstl': f"{loss_distill.item():.3f}",
            'L1': f"{loss_l1.item():.3f}",
            'Spc': f"{loss_spectral.item():.3f}",
            'Prs': f"{loss_pearson.item():.3f}"
        })

        # 🔥 DỌN RÁC VRAM: Ngăn chặn rò rỉ bộ nhớ sau mỗi batch
        del ppg_1d, ecg_1d, ppg_st_2d, ecg_st_2d, ppg_embed, ecg_embed, recon_ecg_1d
        del loss_distill, loss_l1, loss_spectral, loss_pearson, total_loss, loss

    num_batches = len(dataloader)
    return {k: v / num_batches for k, v in metrics.items()}

if __name__ == "__main__":
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Khởi động trên thiết bị: {device}")

    train_dataset = LoadData('processed_data/mimic3_v1_2400_train.npz')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    model = PPG2ECG_Model().to(device)
    
    criterion_dict = {
        'distill': SoftTargetDistillationLoss().to(device),
        'l1': nn.L1Loss().to(device),
        'spectral': SpectralLoss().to(device),
        'pearson': PearsonCorrelationLoss().to(device)
    }

    loss_weights = {
        'distill': WEIGHT_DISTILL, 
        'l1': WEIGHT_L1, 
        'spectral': WEIGHT_SPECTRAL,
        'pearson': WEIGHT_PEARSON
    }

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scaler = GradScaler()
    
    best_loss = float('inf')
    history = {
        'Total Loss': [], 'Distillation Loss': [], 'L1 Loss': [], 
        'Spectral Loss': [], 'Pearson Loss': []
    }

    print("\nBẮT ĐẦU HUẤN LUYỆN...")
    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()

        avg_losses = train_epoch_combined(
            model, train_loader, optimizer, criterion_dict, loss_weights, device, scaler
        )
        
        history['Total Loss'].append(avg_losses['total'])
        history['Distillation Loss'].append(avg_losses['distill'])
        history['L1 Loss'].append(avg_losses['l1'])
        history['Spectral Loss'].append(avg_losses['spectral'])
        history['Pearson Loss'].append(avg_losses['pearson'])
        
        print(f"Epoch [{epoch:03d}/{NUM_EPOCHS}] - Time: {time.time() - start_time:.1f}s")
        print(f"Loss -> Total: {avg_losses['total']:>5.3f} | Dstl: {avg_losses['distill']:>5.3f} | L1: {avg_losses['l1']:>5.3f} | Spc: {avg_losses['spectral']:>5.3f} | Prs: {avg_losses['pearson']:>5.3f}")

        if avg_losses['total'] < best_loss:
            best_loss = avg_losses['total']
            torch.save(model.state_dict(), "PPG2ECG_SOTA_best_model.pth")
            print(f" ⭐ Đã lưu Model tốt nhất tại Epoch {epoch}")
        print("-" * 60)
        
    print("\n🎉 HUẤN LUYỆN HOÀN TẤT!")
    plot_losses_combined(history, title="Training Loss Components Over Epochs")