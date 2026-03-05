from CLIP import ECGDecoder_UNet, ECGEssembleCLIP, FullModelWrapper
from loss import FastSoftCLIPLoss
import torch
from torch.utils.data import DataLoader
from load_data import LoadData, PPG2ECG_Dataset, quality_aware_collate_fn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import numpy as np
import random
import torch.nn.functional as F
from utils import detect_ecg_features
from torchinfo import summary
import shutil
import psutil
# --- (CONSTANTS) ---
SEED = 44
INPUT_LENGTH = 2048
NUM_EPOCHS = 200
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
WEIGHT_CONTRASTIVE = 0.1
WEIGHT_L1 = 1.0      
WEIGHT_PEARSON = 0.5 
WEIGHT_FREQUENCY = 0.01
WEIGHT_MASK = 1.0    # 🔥 TRỌNG SỐ MỚI CHO MASK LOSS
OUTPUT_EMBED_DIM = 128
frequency = 125
def check_system_resources():
    # RAM Hệ thống
    vm = psutil.virtual_memory()
    # Ổ cứng (để lưu file .pth)
    total, used, free = shutil.disk_usage("/")

    print("\n" + "-"*20 + " SYSTEM RESOURCES " + "-"*20)
    print(f"System RAM: {vm.percent}% used ({vm.available / 1024**3:.2f} GB free)")
    print(f"Disk Space: {free / 1024**3:.2f} GB free")
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def plot_losses_combined(losses_dict, title='Training Losses'):
    epochs = range(1, len(next(iter(losses_dict.values()))) + 1)
    plt.figure(figsize=(10, 6))
    
    for label, loss_values in losses_dict.items():
        plt.plot(epochs, loss_values, label=label)
        
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{title.lower().replace(' ', '_')}.png")

# =====================================================================
# 🔥 THÊM MỚI: BỘ LỌC NHIỄU THỜI GIAN (TEMPORAL NOISE FILTER)
# =====================================================================
class TemporalNoiseFilter(torch.nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.attention = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, 16, kernel_size=7, padding=3),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(),
            torch.nn.Conv1d(16, 1, kernel_size=1)
            # 🔥 ĐÃ XÓA nn.Sigmoid() Ở ĐÂY
        )

    def forward(self, x):
        logits = self.attention(x)             # Đầu ra thô (Logits) chạy từ -inf đến +inf
        pred_mask = torch.sigmoid(logits)      # Ép về 0-1 để làm mặt nạ nhân với tín hiệu
        masked_x = x * pred_mask 
        
        # TRẢ VỀ THÊM logits để đưa vào hàm Loss
        return masked_x, pred_mask, logits
# --- PEARSON CORRELATION LOSS ---
class PearsonCorrelationLoss(torch.nn.Module):
    def __init__(self):
        super(PearsonCorrelationLoss, self).__init__()

    def forward(self, x, y):
        x_flat = x.view(x.shape[0], -1)
        y_flat = y.view(y.shape[0], -1)
        
        mean_x = torch.mean(x_flat, dim=1, keepdim=True)
        mean_y = torch.mean(y_flat, dim=1, keepdim=True)
        
        xm = x_flat - mean_x
        ym = y_flat - mean_y
        
        r_num = torch.sum(xm * ym, dim=1)
        r_den = torch.sqrt(torch.sum(xm ** 2, dim=1) * torch.sum(ym ** 2, dim=1) + 1e-8)
        
        r = r_num / r_den
        return 1 - r 

def check_gpu_memory(epoch):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024**2)
        reserved = torch.cuda.memory_reserved(0) / (1024**2)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        
        print(f"\n--- GPU Memory Report (Epoch {epoch}) ---")
        print(f"Used: {allocated:.2f} MB")
        print(f"Reserved (Cache): {reserved:.2f} MB")
        print(f"Free: {total_vram - allocated:.2f} MB / {total_vram:.2f} MB")

def train_epoch_combined(
    noise_filter, model_clip, model_converter, dataloader, optimizer, 
    contrast_loss_fn, mse_loss_fn, pearson_loss_fn, 
    device, scaler,
    loss_weights
):
    noise_filter.train()
    model_clip.train()
    model_converter.train()

    metrics = {k: 0.0 for k in ['total', 'contrast', 'mse', 'pearson', 'mask']}

    progress_bar = tqdm(dataloader, desc="Training (Multi-task)", unit="batch")
    
    for ecg, ppg, _, ppg_sqm, ppg_sqi, ecg_sqi in progress_bar:
        ecg_target = ecg.to(device).float().unsqueeze(1)
        ppg_input = ppg.to(device).float().unsqueeze(1)
        print("shape of ppg_input: ", ppg_input.shape)
        # Đưa Mask thực tế lên GPU (Shape: [Batch, 1, Seq_Len])
        true_ppg_sqm = ppg_sqm.to(device).float().unsqueeze(1)
        
        # ecg_sqi vẫn là 1 số vô hướng cho mỗi mẫu, view thành [Batch, 1, 1] 
        # để chuẩn bị broadcast nhân với true_ppg_sqm
        ecg_sqi = ecg_sqi.to(device).float().view(-1, 1, 1)
        
        # ppg_sqi (vô hướng) KHÔNG CẦN NỮA VÌ ĐÃ CÓ true_ppg_sqm
        # ppg_sqi = ppg_sqi.to(device).float().view(-1, 1, 1) 
        
        optimizer.zero_grad()
        
        # ======================================================
        # 🔥 ĐIỂM THAY ĐỔI CỐT LÕI: TÍNH CONFIDENCE WEIGHTS ĐỘNG
        # ======================================================
        # confidence_weights bây giờ là một ma trận [Batch, 1, Seq_Len]
        # Tại mỗi điểm thời gian, trọng số = Chất lượng PPG tại điểm đó (0 hoặc 1) * Chất lượng tổng thể của ECG
        confidence_weights = true_ppg_sqm * ecg_sqi
        
        with autocast():
            masked_ppg, pred_mask, mask_logits = noise_filter(ppg_input)
            mask_loss = F.binary_cross_entropy_with_logits(mask_logits, true_ppg_sqm)
            
            # ======================================================
            # BƯỚC 2: CHẠY MÔ HÌNH CHÍNH (Với đầu vào đã được làm sạch)
            # ======================================================
            logits_per_ecg, ppg_embedding, feature_lists_PPG = model_clip(ecg_target, masked_ppg)
            c_loss = contrast_loss_fn(logits_per_ecg, ecg_target)
            
            predicted_ecg = model_converter(ppg_embedding, feature_lists_PPG) 

            # ======================================================
            # BƯỚC 3: TÍNH TOÁN LOSS
            # ======================================================
            # unweighted_mse có shape [Batch, 1, Seq_Len]
            unweighted_mse = mse_loss_fn(predicted_ecg, ecg_target) 
            
            # Nhân với confidence_weights [Batch, 1, Seq_Len]
            # Điểm nhiễu sẽ bị nhân với 0, triệt tiêu loss
            m_loss = torch.sum(unweighted_mse * confidence_weights) / (torch.sum(confidence_weights) + 1e-8)

            # Pearson Loss vẫn tính trung bình trên cả đoạn tín hiệu
            # Vì pearson đã trả về shape [Batch], ta nhân với ecg_sqi và trung bình của true_ppg_sqm
            unweighted_pearson = pearson_loss_fn(predicted_ecg, ecg_target)
            
            # Tạo trọng số vô hướng cho mỗi batch để nhân với Pearson Loss
            # (Lấy trung bình của true_ppg_sqm cho mẫu đó * ecg_sqi)
            pearson_weights = true_ppg_sqm.mean(dim=-1).view(-1) * ecg_sqi.view(-1)
            p_loss = torch.sum(unweighted_pearson * pearson_weights) / (torch.sum(pearson_weights) + 1e-8)

            # Tổng hợp toàn bộ Loss
            total_loss = (loss_weights['contrast'] * c_loss + 
                          loss_weights['mse'] * m_loss + 
                          loss_weights['pearson'] * p_loss +
                          loss_weights['mask'] * mask_loss)
        
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        metrics['total'] += total_loss.item()
        metrics['contrast'] += c_loss.item() * loss_weights['contrast']
        metrics['mse'] += m_loss.item() * loss_weights['mse']
        metrics['pearson'] += p_loss.item() * loss_weights['pearson']
        metrics['mask'] += mask_loss.item() * loss_weights['mask']

        progress_bar.set_postfix({
            'Total': f"{total_loss.item():.4f}", 
            'MSE': f"{m_loss.item():.4f}",
            'Pears': f"{p_loss.item():.4f}",
            'Mask': f"{mask_loss.item():.4f}" 
        })

    num_batches = len(dataloader)
    avg_losses = {k: v / num_batches for k, v in metrics.items()}
    return avg_losses

if __name__ == "__main__":
    set_seed(SEED)
    check_system_resources()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = PPG2ECG_Dataset('processed_data/mimic3_v1_train.npz') 
    sample_data = train_dataset[0]
    ecg_signal = sample_data[0]
    length_ecg = len(ecg_signal)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                               shuffle=True, num_workers=4,
                                 pin_memory=True, collate_fn=quality_aware_collate_fn)

    # Khởi tạo thêm Bộ lọc nhiễu
    noise_filter = TemporalNoiseFilter(in_channels=1).to(device)
    model_clip = ECGEssembleCLIP(embed_dim=OUTPUT_EMBED_DIM).to(device)
    model_converter = ECGDecoder_UNet(bottleneck_channels=length_ecg).to(device)

    contrast_loss = FastSoftCLIPLoss(teacher_temp=0.05, student_temp=0.07).to(device)
    mse_loss = torch.nn.MSELoss(reduction='none').to(device) 
    pearson_loss = PearsonCorrelationLoss().to(device)
    
    # Gom toàn bộ parameter vào Optimizer
    params_to_optimize = (list(noise_filter.parameters()) + 
                          list(model_clip.parameters()) + 
                          list(model_converter.parameters()))
    optimizer = AdamW(params_to_optimize, lr=LEARNING_RATE, weight_decay=1e-4)

    loss_weights = {
        'contrast': WEIGHT_CONTRASTIVE, 
        'mse': WEIGHT_L1, 
        'pearson': WEIGHT_PEARSON,
        'mask': WEIGHT_MASK
    }

    scaler = GradScaler()
    best_loss = float('inf')
    
    history = {
        'Total Loss': [], 'Contrastive Loss': [], 'MSE Loss': [], 
        'Pearson Loss': [], 'Mask Loss': []
    }

    print("\n" + "="*30 + " MODEL SUMMARY " + "="*30)


    full_model = FullModelWrapper(noise_filter, model_clip, model_converter)

    # Kiểm tra với input_length = 2048
    # Định dạng: (Batch, Channels, Length)
    print("\n summeray  FullModelWrapper ******************************************************************")
    summary(full_model, input_data=[torch.randn(648, 1, 2400).to(device), 
                                    torch.randn(648, 1, 2400).to(device)])
    print("   ******************************************************************\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        start = time.time()

        avg_losses = train_epoch_combined(
            noise_filter, model_clip, model_converter, train_loader, optimizer,
            contrast_loss, mse_loss, pearson_loss, 
            device, scaler, loss_weights
        )
        history['Total Loss'].append(avg_losses['total'])
        history['Contrastive Loss'].append(avg_losses['contrast'])
        history['MSE Loss'].append(avg_losses['mse'])
        history['Pearson Loss'].append(avg_losses['pearson'])
        history['Mask Loss'].append(avg_losses['mask'])
        
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

        if avg_losses['total'] < best_loss:
            best_loss = avg_losses['total']
            torch.save(noise_filter.state_dict(), "multitask_noise_filter_best.pth")
            torch.save(model_clip.state_dict(), "multitask_clip_best_model.pth")
            torch.save(model_converter.state_dict(), "multitask_decoder_best_model.pth")
            print(f">> Saved best model at epoch {epoch}")

        print(f"Epoch time: {time.time() - start:.1f}s")
        check_gpu_memory(epoch)

        
    print("\nTraining Complete.")
    plot_losses_combined(history, title="Training Loss Components Over Epochs")