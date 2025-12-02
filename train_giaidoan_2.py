import torch
from torch import nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.optim import AdamW
import time
from CLIP import ECGEssembleCLIP, PPGtoECGConverter
from load_data import LoadData
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, List, Any, Tuple # <-- Thêm import

# --- Giả định các import này hoạt động ---
# (Các file này phải tồn tại trong thư mục của bạn)
from ResNet50 import ResNet50_1D
from VisionTransformer import SignalTransformer

# --- Các hằng số ---
OUTPUT_EMBED_DIM = 128 
INPUT_LENGTH = 2400

def load_clip_encoder_weights(converter_model, clip_model_path):
    """
    Tải trọng số PPG Encoder từ file CLIP vào mô hình Converter.
    """
    print(f"Đang tải trọng số Giai đoạn 1 từ: {clip_model_path}")
    clip_state_dict = torch.load(clip_model_path)
    
    encoder_state_dict = {}
    
    # Tên của PPG Encoder trong ECGEssembleCLIP là 'encode_ecg_transformer'
    clip_prefix = "encode_ecg_transformer."
    
    # Tên của PPG Encoder trong PPGtoECGConverter là 'ppg_encoder'
    new_prefix = "ppg_encoder."

    for key, value in clip_state_dict.items():
        if key.startswith(clip_prefix):
            # Đổi tên key:
            # "encode_ecg_transformer.layer1.weight" -> "ppg_encoder.layer1.weight"
            new_key = key.replace(clip_prefix, new_prefix, 1)
            encoder_state_dict[new_key] = value
            
    # Tải các trọng số đã lọc vào mô hình converter
    # strict=False nghĩa là không báo lỗi nếu thiếu trọng số (ví dụ: của decoder)
    converter_model.load_state_dict(encoder_state_dict, strict=False)
    print("Đã tải thành công trọng số PPG Encoder vào mô hình Converter.")

def train_epoch_converter(model, dataloader, optimizer, criterion, device, scaler):
    model.train()
    # (Quan trọng) Giữ encoder ở chế độ eval() (tắt dropout, etc.)
    # vì nó đã được đóng băng.
    model.ppg_encoder.eval() 
    
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training (Phase 2)", unit="batch")
    
    # for ecg, ppg, _,_ in progress_bar: # Không cần 'labels'
    for ecg, ppg, _ in progress_bar: # Không cần 'labels'

        ppg_input = ppg.to(device).float().unsqueeze(1)
        ecg_target = ecg.to(device).float().unsqueeze(1) # Đây là mục tiêu

        optimizer.zero_grad()
        with autocast():
            predicted_ecg = model(ppg_input)
            loss = criterion(predicted_ecg, ecg_target) # Dùng L1 hoặc MSE Loss
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        
    torch.cuda.empty_cache()
    return total_loss / len(dataloader)

def evaluate_converter(model, dataloader, criterion, device):
    model.eval() # Đặt toàn bộ mô hình ở chế độ eval
    total_loss = 0
    with torch.no_grad():
        # for ecg, ppg, _ , _ in dataloader:
        for ecg, ppg, _ in dataloader:

            ppg_input = ppg.to(device).float().unsqueeze(1)
            ecg_target = ecg.to(device).float().unsqueeze(1)

            with autocast():
                predicted_ecg = model(ppg_input)
                loss = criterion(predicted_ecg, ecg_target)
            total_loss += loss.item()
            
    torch.cuda.empty_cache()
    return total_loss / len(dataloader)

# --- HÀM PLOT ---
def plot_losses(train_losses, val_losses, title='Training and Validation Loss'):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b', label='Training Loss')
    plt.plot(epochs, val_losses, 'r', label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    
    # --- CẤU HÌNH ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")
    BATCH_SIZE = 128
    # --- TẢI DATA ---
    print("Đang tải dataset...")
    train_dataset = LoadData('/home/linhhima/Pre_processing_data/datasets/MIMIC_InfoNCE_train.npz')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True) # Tối ưu

    val_dataset = LoadData('/home/linhhima/Pre_processing_data/datasets/MIMIC_InfoNCE_val.npz')
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # ===================================================================
    # --- GIAI ĐOẠN 2: HUẤN LUYỆN CONVERTER ---
    # ===================================================================
    print("\n\n--- BẮT ĐẦU GIAI ĐOẠN 2: HUẤN LUYỆN CONVERTER (PPG-to-ECG) ---")
    
    PHASE_2_EPOCHS = 50 # Cần nhiều epoch hơn để finetune
    PHASE_2_LR = 1e-3   # Tốc độ học cao hơn cho decoder
    CONVERTER_MODEL_PATH = "ppg_to_ecg_converter_best_InfoNCE.pth"
    CLIP_MODEL_PATH = "ecg_ppg_clip_best_model_InfoNCE.pth" 
    # 1. Khởi tạo mô hình Converter
    converter_model = PPGtoECGConverter(
        embed_dim=OUTPUT_EMBED_DIM, 
        input_length=INPUT_LENGTH
    ).to(device)
    
    # 2. Loss cho Giai đoạn 2: L1 (MAE) tốt cho tín hiệu, ít nhạy cảm với nhiễu
    # Thử với MSE loss
    converter_criterion = nn.MSELoss()
    # converter_criterion = nn.L1Loss()


    # 3. Tải trọng số từ Giai đoạn 1
    load_clip_encoder_weights(converter_model, CLIP_MODEL_PATH)

    # 4. Đóng băng (FREEZE) PPG Encoder
    for param in converter_model.ppg_encoder.parameters():
        param.requires_grad = False
    print("Đã đóng băng (freeze) PPG Encoder. Chỉ huấn luyện ECG Decoder.")

    # 5. Optimizer chỉ học các tham số của Decoder
    converter_optimizer = AdamW(converter_model.ecg_decoder.parameters(), lr=PHASE_2_LR)
    
    # 6. Scaler mới cho Giai đoạn 2
    converter_scaler = GradScaler()

    # 7. Vòng lặp huấn luyện mới
    best_converter_loss = float('inf')
    converter_train_losses = []
    converter_val_losses = []

    for epoch in range(1, PHASE_2_EPOCHS + 1):
        start = time.time()

        # Gọi các hàm huấn luyện/đánh giá MỚI
        train_loss = train_epoch_converter(
            converter_model, train_loader, converter_optimizer, 
            converter_criterion, device, converter_scaler
        )
        val_loss = evaluate_converter(
            converter_model, val_loader, converter_criterion, device
        )

        converter_train_losses.append(train_loss)
        converter_val_losses.append(val_loss)

        print(f"\nEpoch {epoch}/{PHASE_2_EPOCHS} (Phase 2)")
        print(f"Train Loss (L1): {train_loss:.6f}")
        print(f"Val Loss (L1):   {val_loss:.6f}")

        if val_loss < best_converter_loss:
            best_converter_loss = val_loss
            torch.save(converter_model.state_dict(), CONVERTER_MODEL_PATH)
            print(f">> Đã lưu model Giai đoạn 2 tốt nhất: {CONVERTER_MODEL_PATH}")

        print(f"Epoch time: {time.time() - start:.1f}s")
        torch.cuda.empty_cache()

    print("\nHoàn tất Giai đoạn 2.")
    plot_losses(converter_train_losses, converter_val_losses, title="Phase 2 - Converter (PPG-to-ECG) Loss")

    print("\n--- HUẤN LUYỆN HOÀN TẤT ---")