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
from typing import Dict, List, Any, Tuple

# --- Giả định các import này hoạt động ---
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
    if not torch.cuda.is_available():
        clip_state_dict = torch.load(clip_model_path, map_location=torch.device('cpu'))
    else:
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
    converter_model.load_state_dict(encoder_state_dict, strict=False)
    print("Đã tải thành công trọng số PPG Encoder vào mô hình Converter.")

def train_epoch_converter(model, dataloader, optimizer, criterion, device, scaler):
    model.train()
    # (Quan trọng) Giữ encoder ở chế độ eval() vì nó đã được đóng băng.
    model.ppg_encoder.eval() 
    
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training (Phase 2)", unit="batch")
    
    for ecg, ppg, _,_ in progress_bar: # Không cần 'labels' hoặc groupID

        ppg_input = ppg.to(device).float().unsqueeze(1)
        ecg_target = ecg.to(device).float().unsqueeze(1) # Đây là mục tiêu

        optimizer.zero_grad()
        with autocast():
            predicted_ecg = model(ppg_input)
            loss = criterion(predicted_ecg, ecg_target) 
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        
        # Cập nhật hiển thị loss ngay trên thanh progress bar
        progress_bar.set_postfix({'loss': loss.item()})
        
    torch.cuda.empty_cache()
    return total_loss / len(dataloader)

# --- HÀM PLOT ĐÃ SỬA (CHỈ VẼ TRAINING LOSS) ---
def plot_losses(train_losses, title='Training Loss'):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b', label='Training Loss')
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

    # --- TẢI DATA (CHỈ TẢI TRAIN) ---
    print("Đang tải dataset...")
    # Chỉ giữ lại tập Train
    train_dataset = LoadData('datasets/normal_train.npz')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # ===================================================================
    # --- GIAI ĐOẠN 2: HUẤN LUYỆN CONVERTER (PPG-to-ECG) ---
    # ===================================================================
    print("\n\n--- BẮT ĐẦU GIAI ĐOẠN 2: HUẤN LUYỆN CONVERTER (CHỈ TRAIN) ---")
    
    PHASE_2_EPOCHS = 50 
    PHASE_2_LR = 1e-3    
    CONVERTER_MODEL_PATH = "ppg_to_ecg_converter_kullback_best_train_loss.pth" # Đổi tên file để phản ánh việc lưu theo train loss
    CLIP_MODEL_PATH = "ecg_ppg_clip_kullback_model.pth" 

    # 1. Khởi tạo mô hình Converter
    converter_model = PPGtoECGConverter(
        embed_dim=OUTPUT_EMBED_DIM, 
        input_length=INPUT_LENGTH
    ).to(device)
    
    # 2. Loss: MSE hoặc L1
    converter_criterion = nn.MSELoss()
    # converter_criterion = nn.L1Loss()

    # 3. Tải trọng số từ Giai đoạn 1 (CLIP)
    try:
        load_clip_encoder_weights(converter_model, CLIP_MODEL_PATH)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {CLIP_MODEL_PATH}. Hãy chắc chắn bạn đã train Phase 1.")
        exit()

    # 4. Đóng băng (FREEZE) PPG Encoder
    # for param in converter_model.ppg_encoder.parameters():
    #     param.requires_grad = False
    # print("Đã đóng băng (freeze) PPG Encoder. Chỉ huấn luyện ECG Decoder.")

    # 5. Optimizer
    converter_optimizer = AdamW(converter_model.ecg_decoder.parameters(), lr=PHASE_2_LR)
    
    # 6. Scaler
    converter_scaler = GradScaler()

    # 7. Vòng lặp huấn luyện (CHỈ TRAIN)
    best_train_loss = float('inf') # Theo dõi train loss tốt nhất thay vì val loss
    converter_train_losses = []

    for epoch in range(1, PHASE_2_EPOCHS + 1):
        start = time.time()

        # Chỉ gọi hàm train
        train_loss = train_epoch_converter(
            converter_model, train_loader, converter_optimizer, 
            converter_criterion, device, converter_scaler
        )

        converter_train_losses.append(train_loss)

        print(f"\nEpoch {epoch}/{PHASE_2_EPOCHS}")
        print(f"Train Loss: {train_loss:.6f}")

        # Logic lưu model: Lưu nếu Training Loss giảm
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(converter_model.state_dict(), CONVERTER_MODEL_PATH)
            print(f">> Đã lưu model (New Best Train Loss): {CONVERTER_MODEL_PATH}")
        
        # Tùy chọn: Lưu model mới nhất mỗi epoch để backup
        # torch.save(converter_model.state_dict(), "ppg_to_ecg_converter_latest.pth")

        print(f"Epoch time: {time.time() - start:.1f}s")
        torch.cuda.empty_cache()

    print("\nHoàn tất Training.")
    plot_losses(converter_train_losses, title="Phase 2 - Converter Training Loss")

    print("\n--- HUẤN LUYỆN HOÀN TẤT ---")