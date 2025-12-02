import torch
from torch import nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.optim import AdamW
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler

# --- Giả định các import này hoạt động ---
# (Các file này phải tồn tại trong thư mục của bạn)
try:
    from CLIP import ECGEssembleCLIP, PPGtoECGConverter
    from load_data import LoadData
    # Các import bên dưới là từ file thứ 2, có thể bạn đã
    # định nghĩa chúng trong file CLIP.py
    from ResNet50 import ResNet50_1D
    from VisionTransformer import SignalTransformer
except ImportError:
    print("CẢNH BÁO: Không thể import 'CLIP', 'load_data' hoặc các mô hình.")
    print("Vui lòng đảm bảo các file .py đó tồn tại trong thư mục làm việc.")
    # Tạo các lớp giả (dummy) để code có thể chạy
    class DummyModel(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__()
        def forward(self, *args, **kwargs): return [torch.randn(16, 16), torch.randn(16, 16)]
    class DummyConverter(nn.Module):
        def __init__(self, *args, **kwargs): 
            super().__init__()
            self.ppg_encoder = nn.Linear(10, 10)
            self.ecg_decoder = nn.Linear(10, 10)
        def forward(self, *args, **kwargs): return torch.randn(16, 1, 2400)
    class DummyDataset(Dataset):
        def __len__(self): return 100
        def __getitem__(self, idx): return torch.randn(2400), torch.randn(2400), torch.tensor(idx % 5), torch.tensor(idx % 2)
    ECGEssembleCLIP = DummyModel
    PPGtoECGConverter = DummyConverter
    LoadData = DummyDataset
    ResNet50_1D = nn.Module
    SignalTransformer = nn.Module


# --- Các hằng số ---
OUTPUT_EMBED_DIM = 128 
INPUT_LENGTH = 2400

# --- Hàm SupCon Loss ---
def supcon_loss(similarity_matrix, labels, temperature=0.07):
    """
    Tính toán SupCon loss (phiên bản hàm độc lập).
    """
    batch_size = labels.shape[0]
    
    # 1. Tạo "mặt nạ dương tính" (positive_mask)
    labels_col = labels.view(batch_size, 1)
    labels_row = labels.view(1, batch_size)
    positive_mask = (labels_col == labels_row).float()
    
    # 2. Điều chỉnh ma trận tương đồng với nhiệt độ
    logits = similarity_matrix / temperature
    
    # 3. Tính loss bằng cách neo vào ECG (loss theo hàng)
    log_probs_row = F.log_softmax(logits, dim=1)
    sum_log_probs_row = (log_probs_row * positive_mask).sum(dim=1)
    num_positives_row = positive_mask.sum(dim=1).clamp(min=1)
    mean_log_probs_row = sum_log_probs_row / num_positives_row
    loss_row = - mean_log_probs_row.mean()
    
    # 4. Tính loss bằng cách neo vào PPG (loss theo cột)
    log_probs_col = F.log_softmax(logits, dim=0)
    sum_log_probs_col = (log_probs_col * positive_mask).sum(dim=0)
    num_positives_col = positive_mask.sum(dim=0).clamp(min=1)
    mean_log_probs_col = sum_log_probs_col / num_positives_col
    loss_col = - mean_log_probs_col.mean()
    
    # 5. Loss cuối cùng là trung bình của cả hai
    loss = (loss_row + loss_col) / 2
    return loss

# --- Vòng lặp huấn luyện và đánh giá (ĐÃ KẾT HỢP) ---

def train_epoch_combined(
    model_clip, model_converter, dataloader, 
    optimizer, criterion_mse, device, scaler, 
    loss_weight_contrastive=1.0, loss_weight_mse=1.0
):
    model_clip.train()
    model_converter.train()
    
    total_loss_all = 0
    total_loss_contrast = 0
    total_loss_mse = 0
    
    progress_bar = tqdm(dataloader, desc="Training (Multi-task)", unit="batch")
    
    for ecg, ppg, labels, groupID in progress_bar:
        ecg_target = ecg.to(device).float().unsqueeze(1)
        ppg_input = ppg.to(device).float().unsqueeze(1)
        groupID_labels = groupID.to(device)
        
        optimizer.zero_grad()

        # Mixed Precision Forward
        with autocast():
            # --- 1. Tác vụ Tương phản (Contrastive Task) ---
            logits_per_ecg, logits_per_ppg = model_clip(ecg_target, ppg_input)
            contrast_loss = supcon_loss(logits_per_ecg, groupID_labels)
            
            # --- 2. Tác vụ Tái tạo (Reconstruction Task) ---
            predicted_ecg = model_converter(ppg_input)
            mse_loss = criterion_mse(predicted_ecg, ecg_target)
            
            # --- 3. Tổng Loss ---
            total_loss = (loss_weight_contrastive * contrast_loss) + \
                         (loss_weight_mse * mse_loss)

        # Scaled Backward
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss_all += total_loss.item()
        total_loss_contrast += contrast_loss.item()
        total_loss_mse += mse_loss.item()

    torch.cuda.empty_cache()
    
    avg_loss_all = total_loss_all / len(dataloader)
    avg_loss_contrast = total_loss_contrast / len(dataloader)
    avg_loss_mse = total_loss_mse / len(dataloader)
    
    return avg_loss_all, avg_loss_contrast, avg_loss_mse


def evaluate_combined(
    model_clip, model_converter, dataloader, 
    criterion_mse, device,
    loss_weight_contrastive=1.0, loss_weight_mse=1.0
):
    model_clip.eval()
    model_converter.eval()
    
    total_loss_all = 0
    total_loss_contrast = 0
    total_loss_mse = 0

    with torch.no_grad():
        for ecg, ppg, labels, groupID in dataloader:
            ecg_target = ecg.to(device).float().unsqueeze(1)
            ppg_input = ppg.to(device).float().unsqueeze(1)
            groupID_labels = groupID.to(device)

            with autocast():
                # --- 1. Tác vụ Tương phản (Contrastive Task) ---
                logits_per_ecg, logits_per_ppg = model_clip(ecg_target, ppg_input)
                contrast_loss = supcon_loss(logits_per_ecg, groupID_labels)
                
                # --- 2. Tác vụ Tái tạo (Reconstruction Task) ---
                predicted_ecg = model_converter(ppg_input)
                mse_loss = criterion_mse(predicted_ecg, ecg_target)
                
                # --- 3. Tổng Loss ---
                total_loss = (loss_weight_contrastive * contrast_loss) + \
                             (loss_weight_mse * mse_loss)

            total_loss_all += total_loss.item()
            total_loss_contrast += contrast_loss.item()
            total_loss_mse += mse_loss.item()

    torch.cuda.empty_cache()
    
    avg_loss_all = total_loss_all / len(dataloader)
    avg_loss_contrast = total_loss_contrast / len(dataloader)
    avg_loss_mse = total_loss_mse / len(dataloader)
    
    return avg_loss_all, avg_loss_contrast, avg_loss_mse

# --- Hàm vẽ biểu đồ ---
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
    plt.savefig(f"{title.lower().replace(' ', '_')}.png")
    plt.show()

# --- HÀM MAIN THỰC THI ---
if __name__ == '__main__':
    # --- Cấu hình ---
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 128
    # Trọng số cho 2 thành phần loss (có thể điều chỉnh)
    WEIGHT_CONTRASTIVE = 1.0
    WEIGHT_MSE = 1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    # --- Tải Data ---
    # Sử dụng bộ dữ liệu SupCon vì nó chứa groupID
    print("Loading dataset...")
    train_dataset = LoadData('/home/linhhima/Pre_processing_data/datasets/MIMIC_SupCon_train.npz')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    val_dataset = LoadData('/home/linhhima/Pre_processing_data/datasets/MIMIC_SupCon_val.npz')
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # --- Khởi tạo Mô hình ---
    model_clip = ECGEssembleCLIP(embed_dim=OUTPUT_EMBED_DIM).to(device)
    model_converter = PPGtoECGConverter(
        embed_dim=OUTPUT_EMBED_DIM, 
        input_length=INPUT_LENGTH
    ).to(device)
    
    criterion_mse = nn.MSELoss()

    # --- !!! QUAN TRỌNG: CHIA SẺ TRỌNG SỐ (WEIGHT SHARING) !!! ---
    # Giả định:
    # - Tên bộ mã hóa PPG trong `ECGEssembleCLIP` là `encode_ecg_transformer`
    # - Tên bộ mã hóa PPG trong `PPGtoECGConverter` là `ppg_encoder`
    # (Hãy thay đổi tên nếu chúng khác trong file CLIP.py của bạn)
    try:
        model_converter.ppg_encoder = model_clip.encode_ecg_transformer
        print("Đã CHIA SẺ TRỌNG SỐ ppg_encoder của Converter -> encode_ecg_transformer của CLIP.")
    except AttributeError as e:
        print(f"LỖI CHIA SẺ TRỌNG SỐ: {e}")
        print("Vui lòng kiểm tra tên các mô-đun encoder trong file CLIP.py!")
        exit(1)

    # --- Optimizer ---
    # Optimizer sẽ quản lý tất cả các tham số của model_clip
    # (bao gồm cả ppg_encoder đã chia sẻ)
    # VÀ các tham số của bộ giải mã (ecg_decoder) của model_converter
    params_to_optimize = list(model_clip.parameters()) + \
                         list(model_converter.ecg_decoder.parameters())
                         
    optimizer = AdamW(params_to_optimize, lr=LEARNING_RATE, weight_decay=1e-4)

    # --- Scaler ---
    scaler = GradScaler()

    # --- Vòng lặp huấn luyện ---
    best_val_loss = float('inf')
    
    train_losses_all, val_losses_all = [], []
    train_losses_contrast, val_losses_contrast = [], []
    train_losses_mse, val_losses_mse = [], []

    for epoch in range(1, NUM_EPOCHS + 1):
        start = time.time()

        train_loss, train_contrast, train_mse = train_epoch_combined(
            model_clip, model_converter, train_loader, optimizer, 
            criterion_mse, device, scaler,
            WEIGHT_CONTRASTIVE, WEIGHT_MSE
        )
        val_loss, val_contrast, val_mse = evaluate_combined(
            model_clip, model_converter, val_loader, 
            criterion_mse, device,
            WEIGHT_CONTRASTIVE, WEIGHT_MSE
        )

        # Lưu trữ loss
        train_losses_all.append(train_loss)
        val_losses_all.append(val_loss)
        train_losses_contrast.append(train_contrast)
        val_losses_contrast.append(val_contrast)
        train_losses_mse.append(train_mse)
        val_losses_mse.append(val_mse)

        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print(f"  [Train] Total Loss: {train_loss:.6f} | Contrast Loss: {train_contrast:.6f} | MSE Loss: {train_mse:.6f}")
        print(f"  [Val]   Total Loss: {val_loss:.6f} | Contrast Loss: {val_contrast:.6f} | MSE Loss: {val_mse:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Lưu cả hai mô hình
            torch.save(model_clip.state_dict(), "multitask_clip_best_model.pth")
            # Chỉ cần lưu bộ giải mã, vì bộ mã hóa đã được lưu trong model_clip
            torch.save(model_converter.ecg_decoder.state_dict(), "multitask_decoder_best_model.pth")
            print(f">> Đã lưu mô hình tốt nhất (Epoch {epoch})")

        print(f"Epoch time: {time.time() - start:.1f}s")
        torch.cuda.empty_cache()
        
    print("\nTraining Complete.")
    
    # --- Vẽ biểu đồ ---
    plot_losses(train_losses_all, val_losses_all, title="Multitask - Total Loss")
    plot_losses(train_losses_contrast, val_losses_contrast, title="Multitask - Contrastive Loss")
    plot_losses(train_losses_mse, val_losses_mse, title="Multitask - MSE Loss")