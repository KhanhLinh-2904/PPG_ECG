import torch
from torch import nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.optim import AdamW
import time
from CLIP import ECGEssembleCLIP
from load_data import LoadData
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler   # <--- Thêm Mixed Precision

OUTPUT_EMBED_DIM = 128 
INPUT_LENGTH = 2400
def supcon_loss(similarity_matrix, labels, temperature=0.07):
    """
    Tính toán SupCon loss (phiên bản hàm độc lập).
    
    Args:
        similarity_matrix (torch.Tensor): Ma trận tương đồng
            kích thước (B, B), ví dụ: (ecg_features @ ppg_features.T).
            similarity_matrix[i, j] là độ tương đồng giữa ECGi và PPGj.
        labels (torch.Tensor): Tensor 1D kích thước (B) chứa
            ID nhóm (groupID) cho mỗi mẫu.
        temperature (float): Hằng số nhiệt độ (tau) để điều chỉnh
                             phân phối.
            
    Returns:
        torch.Tensor: Giá trị loss (vô hướng).
    """
    
    batch_size = labels.shape[0]
    
    # 1. Tạo "mặt nạ dương tính" (positive_mask)
    # positive_mask[i, j] = 1 nếu labels[i] == labels[j]
    # Điều này có nghĩa là ECGi và PPGj thuộc cùng một nhóm.
    labels_col = labels.view(batch_size, 1)
    labels_row = labels.view(1, batch_size)
    positive_mask = (labels_col == labels_row).float()
    # 2. Điều chỉnh ma trận tương đồng với nhiệt độ
    logits = similarity_matrix / temperature
    
    # 3. Tính loss bằng cách neo vào ECG (loss theo hàng)
    # Đây là L_i = - (1/|P(i)|) * sum_{p in P(i)} log_prob(i, p)
    # trong đó log_prob(i, p) là log_softmax của hàng i, tại cột p.
    
    # log_softmax được tính trên tất cả các mẫu PPG (dim=1)
    log_probs_row = F.log_softmax(logits, dim=1)
    
    # Chọn ra các log_probs của các cặp dương tính
    sum_log_probs_row = (log_probs_row * positive_mask).sum(dim=1)
    
    # |P(i)| = số lượng mẫu dương tính cho mỗi neo ECG
    num_positives_row = positive_mask.sum(dim=1)
    
    # Xử lý trường hợp một mẫu không có cặp dương tính nào 
    # (mặc dù trong trường hợp này, ít nhất nó cũng dương tính với chính nó)
    num_positives_row = num_positives_row.clamp(min=1)
    
    # Tính trung bình loss cho mỗi neo (hàng)
    # loss_i = - (1/|P(i)|) * sum_p
    mean_log_probs_row = sum_log_probs_row / num_positives_row
    
    # Loss cuối cùng (theo hàng) là trung bình của tất cả các loss_i
    loss_row = - mean_log_probs_row.mean()
    
    # 4. Tính loss bằng cách neo vào PPG (loss theo cột)
    # Tương tự, nhưng tính log_softmax trên các mẫu ECG (dim=0)
    log_probs_col = F.log_softmax(logits, dim=0)
    sum_log_probs_col = (log_probs_col * positive_mask).sum(dim=0)
    num_positives_col = positive_mask.sum(dim=0)
    num_positives_col = num_positives_col.clamp(min=1)
    mean_log_probs_col = sum_log_probs_col / num_positives_col
    loss_col = - mean_log_probs_col.mean()
    
    # 5. Loss cuối cùng là trung bình của cả hai
    loss = (loss_row + loss_col) / 2
    
    return loss

def manual_info_nce_loss(logits):
    """
    Triển khai InfoNCE "thủ công" bằng công thức LogSumExp.
    Hàm này tương đương về mặt toán học với Cách 1.
    """
    N = logits.shape[0]

    # --- Tính loss_ecg (ECG-to-PPG, tính trên các hàng) ---
    
    # 1. Lấy điểm của các cặp "đúng" (đường chéo chính)
    # Đây là các logit s_ii
    positive_scores_ecg = torch.diag(logits)
    
    # 2. Tính mẫu số: log(sum(exp(s_ik))) cho mỗi hàng
    # Dùng logsumexp để tránh tràn số khi tính exp()
    log_sum_exp_ecg = torch.logsumexp(logits, dim=1)
    
    # 3. Tính loss cho mỗi hàng và lấy trung bình
    # L_i = log(sum(exp)) - s_ii
    loss_ecg = (log_sum_exp_ecg - positive_scores_ecg).mean()
    
    # --- Tính loss_ppg (PPG-to-ECG, tính trên các cột) ---
    # Ta chỉ cần chuyển vị (transpose) ma trận logits
    logits_t = logits.t()
    
    positive_scores_ppg = torch.diag(logits_t)
    log_sum_exp_ppg = torch.logsumexp(logits_t, dim=1)
    
    loss_ppg = (log_sum_exp_ppg - positive_scores_ppg).mean()

    return (loss_ecg + loss_ppg) / 2 

def contrastive_loss(logits):
    N = logits.shape[0]
    labels = torch.arange(N, device=logits.device) 
    
    loss_ecg = F.cross_entropy(logits, labels)
    loss_ppg = F.cross_entropy(logits.t(), labels)
    return (loss_ecg + loss_ppg) / 2


def train_epoch(model, dataloader, optimizer, device, scaler):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training", unit="batch")
    
    for ecg, ppg, labels, groupID in progress_bar:
    # for ecg, ppg, labels in progress_bar:

        ecg = ecg.to(device).float().unsqueeze(1)
        ppg = ppg.to(device).float().unsqueeze(1)
        labels = labels.to(device)
        groupID = groupID.to(device)
        optimizer.zero_grad()

        # Mixed Precision Forward
        with autocast():
            logits_per_ecg, logits_per_ppg = model(ecg, ppg)
            # loss = manual_info_nce_loss(logits_per_ecg)
            loss = supcon_loss(logits_per_ecg, groupID)

        # Scaled Backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    torch.cuda.empty_cache()  # <--- Xoá bộ nhớ cache sau mỗi epoch
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for ecg, ppg, labels, groupID in dataloader:
        # for ecg, ppg, labels in dataloader:

            ecg = ecg.to(device).float().unsqueeze(1)
            ppg = ppg.to(device).float().unsqueeze(1)
            groupID = groupID.to(device)
            logits_per_ecg, logits_per_ppg = model(ecg, ppg)
            # loss = manual_info_nce_loss(logits_per_ecg)
            loss = supcon_loss(logits_per_ecg, groupID)
            total_loss += loss.item()

    torch.cuda.empty_cache()
    return total_loss / len(dataloader)


def plot_losses(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b', label='Training Loss')
    plt.plot(epochs, val_losses, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 128   # <--- Bạn có thể thử tăng batch size lên một chút nhờ mixed precision

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading dataset...")
    train_dataset = LoadData('/home/linhhima/Pre_processing_data/datasets/MIMIC_SupCon_train.npz')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = LoadData('/home/linhhima/Pre_processing_data/datasets/MIMIC_SupCon_val.npz')
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = ECGEssembleCLIP(embed_dim=OUTPUT_EMBED_DIM).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    scaler = GradScaler()   # <--- quan trọng

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(1, NUM_EPOCHS + 1):
        start = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, device, scaler)
        val_loss = evaluate(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "ecg_ppg_clip_best_model.pth")
            print(">> Saved new best model")

        print(f"Epoch time: {time.time() - start:.1f}s")
        torch.cuda.empty_cache()
    print("\nTraining Complete.")
    plot_losses(train_losses, val_losses)
