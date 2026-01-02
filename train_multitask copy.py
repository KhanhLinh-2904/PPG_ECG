from CLIP import ECGDecoder_UNet, ECGEssembleCLIP
from Kullback import FastSoftCLIPLoss
import torch
from load_data import LoadData
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from multitask_loss import CombinedLoss

# --- CÁC HẰNG SỐ (CONSTANTS) ---
SEED = 44
NUM_EPOCHS = 100        # Yêu cầu: 100 epoch
LEARNING_RATE = 1e-4    # Tăng LR một chút để hội tụ nhanh hơn với 1 sample
OUTPUT_EMBED_DIM = 128
WEIGHT_CONTRASTIVE = 0.1
WEIGHT_MSE = 1.0        
WEIGHT_PEARSON = 0.5    

def set_seed(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)

# --- ĐỊNH NGHĨA PEARSON LOSS (Để đánh giá độ tương đồng hình dạng) ---
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
        return 1 - torch.mean(r)

if __name__ == "__main__":
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. CHUẨN BỊ DỮ LIỆU (LẤY 1 SAMPLE) ---
    print("Loading dataset to extract 1 sample...")
    full_dataset = LoadData('datasets/normal_train.npz')
    
    # Lấy phần tử đầu tiên (index 0)
    # Giả sử LoadData trả về tuple: (ecg, ppg, label, group, ...)
    sample_data = full_dataset[0] 
    ecg_raw, ppg_raw = sample_data[0], sample_data[1]

    # Chuyển sang Tensor và thêm Batch Dimension [Batch=1, Channel=1, Length]
    # np.array -> Tensor -> Unsqueeze -> Device
    ecg_target = torch.tensor(ecg_raw).float().unsqueeze(0).unsqueeze(0).to(device)
    ppg_input = torch.tensor(ppg_raw).float().unsqueeze(0).unsqueeze(0).to(device)

    print(f"Shape Input PPG: {ppg_input.shape}")
    print(f"Shape Target ECG: {ecg_target.shape}")

    # --- 2. KHỞI TẠO MODEL & LOSS ---
    model_clip = ECGEssembleCLIP(embed_dim=OUTPUT_EMBED_DIM).to(device)
    model_converter = ECGDecoder_UNet(bottleneck_channels=2048).to(device)

    contrast_loss_fn = FastSoftCLIPLoss(teacher_temp=0.05, student_temp=0.07).to(device)
    combined_loss_fn = CombinedLoss(alpha=1.0, beta=1.0).to(device)
    pearson_loss_fn = PearsonCorrelationLoss().to(device)

    params = list(model_clip.parameters()) + list(model_converter.parameters())
    optimizer = AdamW(params, lr=LEARNING_RATE, weight_decay=1e-4)
    scaler = GradScaler()

    history = {'Total': [], 'MSE': [], 'Pearson': []}

    # --- 3. TRAINING LOOP (CHỈ TRÊN 1 SAMPLE) ---
    print("\nStarting Overfitting Test on Single Sample...")
    
    model_clip.train()
    model_converter.train()

    for epoch in range(1, NUM_EPOCHS + 1):
        optimizer.zero_grad()

        with autocast():
            # Forward Pass
            logits_per_ecg, ppg_embedding, feature_lists_PPG = model_clip(ecg_target, ppg_input)
            
            # Reconstruction
            predicted_ecg = model_converter(ppg_embedding, feature_lists_PPG)

            # Calculate Losses
            loss_contrast = contrast_loss_fn(logits_per_ecg, ecg_target) # Có thể lỗi dimension nếu batch=1, nhưng thử xem
            loss_mse = combined_loss_fn(predicted_ecg, ecg_target)
            loss_pearson = pearson_loss_fn(predicted_ecg, ecg_target)

            # Tổng hợp Loss
            total_loss = (WEIGHT_CONTRASTIVE * loss_contrast) + \
                         (WEIGHT_MSE * loss_mse) + \
                         (WEIGHT_PEARSON * loss_pearson)

        # Backward
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Log
        history['Total'].append(total_loss.item())
        history['MSE'].append(loss_mse.item())
        history['Pearson'].append(loss_pearson.item())

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{NUM_EPOCHS} | Total: {total_loss.item():.6f} | MSE: {loss_mse.item():.6f} | Pearson: {loss_pearson.item():.6f}")

    # --- 4. VẼ KẾT QUẢ SO SÁNH ---
    print("\nTraining Complete. Plotting results...")
    
    # Chuyển về CPU để vẽ
    model_converter.eval()
    with torch.no_grad():
        _, ppg_emb, feats = model_clip(ecg_target, ppg_input)
        pred_final = model_converter(ppg_emb, feats)
    
    ecg_real_np = ecg_target.squeeze().cpu().numpy()
    ecg_pred_np = pred_final.squeeze().cpu().numpy()
    ppg_np = ppg_input.squeeze().cpu().numpy()

    plt.figure(figsize=(12, 6))
    
    # Plot 1: Loss Curve
    plt.subplot(2, 1, 1)
    plt.plot(history['MSE'], label='MSE Loss')
    plt.plot(history['Pearson'], label='Pearson Loss')
    plt.title('Loss over 100 Epochs (Single Sample)')
    plt.legend()
    plt.grid(True)

    # Plot 2: Signal Comparison
    plt.subplot(2, 1, 2)
    plt.plot(ecg_real_np, label='Ground Truth ECG', color='black', linewidth=1.5)
    plt.plot(ecg_pred_np, label='Predicted ECG', color='red', linestyle='--')
    plt.title(f'Overfit Result (Epoch {NUM_EPOCHS})')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()