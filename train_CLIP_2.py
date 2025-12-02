import torch
from torch import nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.optim import AdamW
import time
from CLIP import ECGEssembleCLIP
from Kullback import SoftCLIPLoss
from load_data import LoadData
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler


OUTPUT_EMBED_DIM = 128 
INPUT_LENGTH = 2400

# --- CÁC HÀM LOSS GIỮ NGUYÊN ---
def supcon_loss(similarity_matrix, labels, temperature=0.07):
    """
    Tính toán SupCon loss.
    """
    batch_size = labels.shape[0]
    labels_col = labels.view(batch_size, 1)
    labels_row = labels.view(1, batch_size)

    positive_mask = (labels_col == labels_row).float()
    logits = similarity_matrix / temperature
    # print("positive_mask: ", positive_mask)
    # Loss theo hàng (Neo vào ECG)
    log_probs_row = F.log_softmax(logits, dim=1)
    sum_log_probs_row = (log_probs_row * positive_mask).sum(dim=1)
    num_positives_row = positive_mask.sum(dim=1).clamp(min=1)
    mean_log_probs_row = sum_log_probs_row / num_positives_row
    loss_row = - mean_log_probs_row.mean()
    
    # Loss theo cột (Neo vào PPG)
    log_probs_col = F.log_softmax(logits, dim=0)
    sum_log_probs_col = (log_probs_col * positive_mask).sum(dim=0)
    num_positives_col = positive_mask.sum(dim=0).clamp(min=1)
    mean_log_probs_col = sum_log_probs_col / num_positives_col
    loss_col = - mean_log_probs_col.mean()
    
    loss = (loss_row + loss_col) / 2
    return loss

def manual_info_nce_loss(logits):
    N = logits.shape[0]
    positive_scores_ecg = torch.diag(logits)
    log_sum_exp_ecg = torch.logsumexp(logits, dim=1)
    loss_ecg = (log_sum_exp_ecg - positive_scores_ecg).mean()
    
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

# --- HÀM TRAIN ---
def train_epoch(model, dataloader, optimizer, device, scaler):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training", unit="batch")
    loss_fn = SoftCLIPLoss(teacher_temp=0.05, student_temp=0.07)
    
    for ecg, ppg, labels, groupID in progress_bar:
        ecg = ecg.to(device).float().unsqueeze(1)
        ppg = ppg.to(device).float().unsqueeze(1)
        # labels = labels.to(device) # Không dùng labels này trong supcon_loss hiện tại
        groupID = groupID.to(device)
        
        optimizer.zero_grad()

        # Mixed Precision Forward
        with autocast():
            logits_per_ecg, logits_per_ppg = model(ecg, ppg)
            # loss = supcon_loss(logits_per_ecg, groupID)
            loss = loss_fn(logits_per_ecg, ecg)


        # Scaled Backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        
        # Cập nhật hiển thị loss trên thanh progress bar
        progress_bar.set_postfix({'loss': loss.item()})

    torch.cuda.empty_cache()
    return total_loss / len(dataloader)

# --- SỬA HÀM PLOT (CHỈ VẼ TRAIN) ---
def plot_losses(train_losses):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b', label='Training Loss')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- MAIN ---
if __name__ == '__main__':
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading training dataset...")
    # Chỉ load file train
    train_dataset = LoadData('datasets/normal_train.npz')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Khởi tạo Model
    model = ECGEssembleCLIP(embed_dim=OUTPUT_EMBED_DIM).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scaler = GradScaler()

    train_losses = []

    print("Start Training (No Validation)...")
    for epoch in range(1, NUM_EPOCHS + 1):
        start = time.time()

        # Chỉ chạy train
        train_loss = train_epoch(model, train_loader, optimizer, device, scaler)
        train_losses.append(train_loss)

        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}")

        # Vì không có validation để kiểm tra model tốt nhất, 
        # ta sẽ lưu model ở mỗi epoch (model mới nhất)
        torch.save(model.state_dict(), "ecg_ppg_clip_kullback_model.pth")
        print(">> Saved latest model")

        print(f"Epoch time: {time.time() - start:.1f}s")
        torch.cuda.empty_cache()

    print("\nTraining Complete.")
    plot_losses(train_losses)