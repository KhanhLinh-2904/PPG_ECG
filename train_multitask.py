from CLIP import ECGDecoder_UNet, ECGEssembleCLIP
from Kullback import SoftCLIPLoss
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
SEED = 44
INPUT_LENGTH = 2400
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
WEIGHT_CONTRASTIVE = 0.1
WEIGHT_L1 = 1.0
OUTPUT_EMBED_DIM = 128

def set_seed(seed_value: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    print(f"Random seed set to: {seed_value}")

def plot_losses(train_losses, title='Training Loss'):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b', label='Training Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{title.lower().replace(' ', '_')}.png")
    plt.show()

def train_epoch_combined(
   model_clip, model_converter, dataloader, optimizer, contrast_loss, 
            mse_loss, device, scaler,
    loss_weight_contrastive=1.0, loss_weight_mse=1.0
):
    model_clip.train()
    model_converter.train()
    
    total_loss_all = 0
    total_loss_contrast = 0
    total_loss_l1 = 0
    progress_bar = tqdm(dataloader, desc="Training (Multi-task)", unit="batch")
    
    for ecg, ppg, labels, groupID, _ in progress_bar:
        ecg_target = ecg.to(device).float().unsqueeze(1)
        ppg_input = ppg.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()

        # Mixed Precision Forward
        with autocast():
            # --- 1. Tác vụ Tương phản (Contrastive Task) ---
            logits_per_ecg, ppg_embedding, feature_lists_PPG = model_clip(ecg_target, ppg_input)
            c_loss = contrast_loss(logits_per_ecg, ecg_target)
            
            # --- 2. Tác vụ Tái tạo (Reconstruction Task) ---
            predicted_ecg = model_converter(ppg_embedding, feature_lists_PPG)
            l_loss = mse_loss(predicted_ecg, ecg_target)
            # --- 3. Tổng Loss ---
            c_loss = loss_weight_contrastive * c_loss
            l_loss = loss_weight_mse * l_loss
            total_loss = c_loss + l_loss 
        # Scaled Backward
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss_all += total_loss.item()
        total_loss_contrast += c_loss.item()
        total_loss_l1 += l_loss.item()

    torch.cuda.empty_cache()
    
    avg_loss_all = total_loss_all / len(dataloader)
    avg_loss_contrast = total_loss_contrast / len(dataloader)
    avg_loss_l1 = total_loss_l1 / len(dataloader)
    
    return avg_loss_all, avg_loss_contrast, avg_loss_l1

if __name__ == "__main__":
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading training dataset...")
    train_dataset = LoadData('datasets/normal_train.npz')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # val_dataset = LoadData('datasets/normal_val.npz')
    # val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    model_clip =  ECGEssembleCLIP(embed_dim=OUTPUT_EMBED_DIM).to(device)
    model_converter = ECGDecoder_UNet(
        bottleneck_channels=2048
    ).to(device)

    contrast_loss = SoftCLIPLoss(teacher_temp=0.05, student_temp=0.07)
    mse_loss = torch.nn.MSELoss()
    params_to_optimize = list(model_clip.parameters()) + \
                         list(model_converter.parameters())
                         
    optimizer = AdamW(params_to_optimize, lr=LEARNING_RATE, weight_decay=1e-4)

    # --- Scaler ---
    scaler = GradScaler()
    best_loss = float('inf')
    train_losses_all = []
    train_losses_contrast = []
    train_losses_l1 = []

    for epoch in range(1, NUM_EPOCHS + 1):
        start = time.time()
        train_loss, train_contrast, train_l1 = train_epoch_combined(
            model_clip, model_converter, train_loader, optimizer, contrast_loss, 
            mse_loss, device, scaler,
            WEIGHT_CONTRASTIVE, WEIGHT_L1
        )
        train_losses_all.append(train_loss)
        train_losses_contrast.append(train_contrast)
        train_losses_l1.append(train_l1)

        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print(f"  [Train] Total Loss: {train_loss:.6f} | Contrast Loss: {train_contrast:.6f} | L1 Loss: {train_l1:.6f}")

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model_clip.state_dict(), "multitask_clip_best_model.pth")
            torch.save(model_converter.state_dict(), "multitask_decoder_best_model.pth")
            print(f">> Đã lưu mô hình tốt nhất (Epoch {epoch})")

        print(f"Epoch time: {time.time() - start:.1f}s")
        torch.cuda.empty_cache()
        
    print("\nTraining Complete.")
    
    # --- Vẽ biểu đồ ---
    plot_losses(train_losses_all, title="Multitask - Total Loss")
    plot_losses(train_losses_contrast, title="Multitask - Contrastive Loss")
    plot_losses(train_losses_l1, title="Multitask - L1 Loss")

