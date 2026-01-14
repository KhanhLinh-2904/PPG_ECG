from CLIP import ECGDecoder_UNet, ECGEssembleCLIP
from Kullback import FastSoftCLIPLoss
import torch
from torch.utils.data import DataLoader
from load_data import LoadData
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from multitask_loss import CombinedLoss

# --- CÁC HẰNG SỐ (CONSTANTS) ---
SEED = 44
INPUT_LENGTH = 2400
NUM_EPOCHS = 200
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
WEIGHT_CONTRASTIVE = 0.1
WEIGHT_L1 = 1.0      
WEIGHT_PEARSON = 0.5  
OUTPUT_EMBED_DIM = 128

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def plot_losses_combined(losses_dict, title='Training Losses'):
    """Vẽ tất cả các thành phần loss trên cùng một biểu đồ để so sánh"""
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
    plt.show()

# --- 1. ĐỊNH NGHĨA PEARSON CORRELATION LOSS ---
class PearsonCorrelationLoss(torch.nn.Module):
    def __init__(self):
        super(PearsonCorrelationLoss, self).__init__()

    def forward(self, x, y):
        # x: Predicted [Batch, 1, Length]
        # y: Target [Batch, 1, Length]
        
        # Flatten về [Batch, Length] để tính toán
        x_flat = x.view(x.shape[0], -1)
        y_flat = y.view(y.shape[0], -1)
        
        # Tính mean cho từng mẫu trong batch
        mean_x = torch.mean(x_flat, dim=1, keepdim=True)
        mean_y = torch.mean(y_flat, dim=1, keepdim=True)
        
        # Center data (trừ mean)
        xm = x_flat - mean_x
        ym = y_flat - mean_y
        
        # Tính tử số (Covariance)
        r_num = torch.sum(xm * ym, dim=1)
        
        # Tính mẫu số (Product of Stds)
        # Thêm 1e-8 để tránh chia cho 0
        r_den = torch.sqrt(torch.sum(xm ** 2, dim=1) * torch.sum(ym ** 2, dim=1) + 1e-8)
        
        # Hệ số Pearson r [-1, 1]
        r = r_num / r_den
        
        # Loss = 1 - r (Tối thiểu hóa: r -> 1 thì Loss -> 0)
        return 1 - torch.mean(r)

def train_epoch_combined(
    model_clip, model_converter, dataloader, optimizer, 
    contrast_loss_fn, mse_loss_fn, pearson_loss_fn, # Thêm tham số loss function
    device, scaler,
    loss_weight_contrastive=1.0, loss_weight_mse=1.0, loss_weight_pearson=0.5 # Thêm tham số trọng số
):
    model_clip.train()
    model_converter.train()
    
    total_loss_all = 0
    total_loss_contrast = 0
    total_loss_mse = 0
    total_loss_pearson = 0 # Biến theo dõi mới

    progress_bar = tqdm(dataloader, desc="Training (Multi-task)", unit="batch")
    
    for ecg, ppg, labels, groupID, _ in progress_bar:
        ecg_target = ecg.to(device).float().unsqueeze(1)
        ppg_input = ppg.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()

        # Mixed Precision Forward
        with autocast():
            # --- 1. Tác vụ Tương phản (Contrastive Task) ---
            logits_per_ecg, ppg_embedding, feature_lists_PPG = model_clip(ecg_target, ppg_input)
            c_loss = contrast_loss_fn(logits_per_ecg, ecg_target)
            
            # --- 2. Tác vụ Tái tạo (Reconstruction Task) ---
            # Lưu ý: model_converter cần nhận cả feature_lists để chạy U-Net Skip Connections
            predicted_ecg = model_converter(ppg_embedding, feature_lists_PPG) 
            
            # Tính MSE Loss
            m_loss = mse_loss_fn(predicted_ecg, ecg_target)

            # Tính Pearson Loss
            # p_loss = pearson_loss_fn(predicted_ecg, ecg_target)

            # --- 3. Tổng Loss (Weighted Sum) ---
            weighted_c_loss = loss_weight_contrastive * c_loss
            weighted_m_loss = loss_weight_mse * m_loss
            # weighted_p_loss = loss_weight_pearson * p_loss
            
            total_loss = weighted_c_loss + weighted_m_loss 
            # total_loss = weighted_c_loss + weighted_m_loss + weighted_p_loss


        
        # Scaled Backward
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Cộng dồn loss để tính trung bình
        total_loss_all += total_loss.item()
        total_loss_contrast += weighted_c_loss.item()
        total_loss_mse += weighted_m_loss.item()
        # total_loss_pearson += weighted_p_loss.item()
        
        # Hiển thị loss realtime trên progress bar
        progress_bar.set_postfix({
            'Total': f"{total_loss.item():.4f}", 
            'Con': f"{weighted_c_loss.item():.4f}",
            'MSE': f"{weighted_m_loss.item():.4f}",
            # 'Pears': f"{weighted_p_loss.item():.4f}"
        })

    torch.cuda.empty_cache()
    
    # Tính trung bình epoch
    avg_loss_all = total_loss_all / len(dataloader)
    avg_loss_contrast = total_loss_contrast / len(dataloader)
    avg_loss_mse = total_loss_mse / len(dataloader)
    # avg_loss_pearson = total_loss_pearson / len(dataloader)
    return avg_loss_all, avg_loss_contrast, avg_loss_mse
    # return avg_loss_all, avg_loss_contrast, avg_loss_mse, avg_loss_pearson

if __name__ == "__main__":
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading training dataset...")
    # Cập nhật đường dẫn dataset của bạn nếu cần
    train_dataset = LoadData('datasets/total_record_mm_train.npz') 
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # Khởi tạo mô hình
    model_clip = ECGEssembleCLIP(embed_dim=OUTPUT_EMBED_DIM).to(device)
    model_converter = ECGDecoder_UNet(bottleneck_channels=2048).to(device)

    # --- KHỞI TẠO LOSS FUNCTIONS ---
    contrast_loss = FastSoftCLIPLoss(teacher_temp=0.05, student_temp=0.07).to(device)
    # combined_loss = CombinedLoss(alpha=1.0, beta=1.0).to(device)
    mse_loss = torch.nn.MSELoss().to(device)
    pearson_loss = PearsonCorrelationLoss().to(device)
    # Optimizer
    params_to_optimize = list(model_clip.parameters()) + list(model_converter.parameters())
    optimizer = AdamW(params_to_optimize, lr=LEARNING_RATE, weight_decay=1e-4)

    # Scaler
    scaler = GradScaler()
    best_loss = float('inf')
    
    # Dictionary lưu lịch sử loss
    history = {
        'Total Loss': [],
        'Contrastive Loss': [],
        'MSE Loss': [],
        # 'Pearson Loss': []
    }

    # --- VÒNG LẶP TRAINING ---
    for epoch in range(1, NUM_EPOCHS + 1):
        start = time.time()
        
        train_loss, train_contrast, train_mse = train_epoch_combined(
            model_clip, model_converter, train_loader, optimizer, 
            contrast_loss, mse_loss, pearson_loss, # Truyền loss function mới
            device, scaler,
            loss_weight_contrastive=WEIGHT_CONTRASTIVE, 
            loss_weight_mse=WEIGHT_L1,
            loss_weight_pearson=WEIGHT_PEARSON # Truyền weight mới
        )
        
        # Lưu lịch sử
        history['Total Loss'].append(train_loss)
        history['Contrastive Loss'].append(train_contrast)
        history['MSE Loss'].append(train_mse)
        # history['Pearson Loss'].append(train_pearson)

        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        # print(f"  [Train] Total: {train_loss:.5f} | Con: {train_contrast:.5f} | MSE: {train_mse:.5f} | Pearson: {train_pearson:.5f}")

        # Lưu model tốt nhất
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model_clip.state_dict(), "multitask_clip_best_model.pth")
            torch.save(model_converter.state_dict(), "multitask_decoder_best_model.pth")
            print(f">> Saved best model at epoch {epoch}")

        print(f"Epoch time: {time.time() - start:.1f}s")
        torch.cuda.empty_cache()
        
    print("\nTraining Complete.")
    
    # Vẽ biểu đồ tổng hợp
    plot_losses_combined(history, title="Training Loss Components Over Epochs")