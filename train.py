import torch
from torch import nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.optim import AdamW
import time
from CLIP import ECGEssembleCLIP
from ResNet50 import ResNet50_1D
from VisionTransformer import SignalTransformer
from load_data import LoadData
import matplotlib.pyplot as plt
OUTPUT_EMBED_DIM = 128 
INPUT_LENGTH = 2400 


def contrastive_loss(logits):
    """
    Tính Symmetric Cross-Entropy Loss cho Contrastive Learning.
    """
    N = logits.shape[0]
    # Target là các vị trí trên đường chéo chính (0, 1, 2, ..., N-1)
    labels = torch.arange(N, device=logits.device) 
    
    # Loss 1: Từ ECG -> PPG 
    loss_ecg = F.cross_entropy(logits, labels)
    
    # Loss 2: Từ PPG -> ECG 
    loss_ppg = F.cross_entropy(logits.t(), labels)
    
    return (loss_ecg + loss_ppg) / 2

# --- HÀM HUẤN LUYỆN (Train Epoch) ---

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training", unit="batch")
    
    for ecg, ppg, _ in progress_bar:
        # Thêm channel dimension (1) và chuyển sang float. Input shape: [N, 1, L]
        ecg, ppg = ecg.to(device).float().unsqueeze(1), ppg.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        
        # Lan truyền thuận
        logits_per_ecg, logits_per_ppg = model(ecg, ppg)
        
        # Tính toán mất mát
        loss = contrastive_loss(logits_per_ecg)
        
        # Lan truyền ngược
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
    return total_loss / len(dataloader)

# --- HÀM ĐÁNH GIÁ (Evaluate) ---

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for ecg, ppg, _ in dataloader:
            ecg, ppg = ecg.to(device).float().unsqueeze(1), ppg.to(device).float().unsqueeze(1)

            logits_per_ecg, logits_per_ppg = model(ecg, ppg)
            loss = contrastive_loss(logits_per_ecg)
            total_loss += loss.item()
            
    return total_loss / len(dataloader)

def plot_losses(train_losses, val_losses):
    """Vẽ biểu đồ Training Loss và Validation Loss."""
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b', label='Training Loss')
    plt.plot(epochs, val_losses, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Symmetric Cross-Entropy)')
    plt.legend()
    plt.grid(True)
    plt.show()
    print("Biểu đồ Loss đã được vẽ thành công.")
# --- KHỐI HUẤN LUYỆN CHÍNH ---

if __name__ == '__main__':
    
    # 0. Thiết lập Siêu tham số (Hyperparameters)
    NUM_EPOCHS = 200
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 128 # Kích thước batch lớn hơn thường cần thiết cho Contrastive Learning

    # 1. Thiết bị (Device Setup)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Tải Dữ liệu
    print("Data processing...")
    try:
        train_dataset = LoadData('datasets/MIMIC_train.npz')
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_dataset = LoadData('datasets/MIMIC_val.npz')
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    except Exception as e:
        print(f"Error with loading datasets: {e}")
      
    # 3. Khởi tạo Mô hình, Optimizer, và Loss
    model = ECGEssembleCLIP(embed_dim=OUTPUT_EMBED_DIM).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Learning parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Model starts with {NUM_EPOCHS} epochs.")

    # 4. Vòng lặp Huấn luyện Chính
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    for epoch in range(1, NUM_EPOCHS + 1):
        # Huấn luyện
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # Đánh giá
        val_loss = evaluate(model, val_loader, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        end_time = time.time()
        epoch_time = end_time - start_time
        minutes = int(epoch_time // 60)
        seconds = int(epoch_time % 60)
        
        print(f"\n--- Epoch {epoch}/{NUM_EPOCHS} ---")
        print(f"Time Taken: {minutes}m {seconds}s")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")

        # Lưu mô hình tốt nhất
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Sử dụng f-string để lưu tên file có versioning
            torch.save(model.state_dict(), f'ecg_ppg_clip_best_model.pth')
            print("Save the best epoch.")
            
    print("\nComplete!.")
    plot_losses(train_losses, val_losses)
