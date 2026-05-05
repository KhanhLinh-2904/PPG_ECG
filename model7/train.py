import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import numpy as np
from load_data import LoadData

# Đảm bảo import đúng các class model mới của bạn
from model import Stage1_ECG_AutoEncoder, Stage2_PPG2ECG_Improved, Stage2LossImproved

# ==========================================
# 0. CÀI ĐẶT THÔNG SỐ (HYPERPARAMETERS)
# ==========================================
BATCH_SIZE = 32
EPOCHS_STAGE_1 = 50
EPOCHS_STAGE_2 = 100
LR_STAGE_1 = 1e-4
LR_STAGE_2 = 1e-4

os.makedirs("saved_models", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang sử dụng thiết bị: {device}")


def get_dataloaders():
    train_loader = DataLoader(LoadData('/home/linhhima/Diffusion datasets/train.npz'), 
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(LoadData('/home/linhhima/Diffusion datasets/val.npz'), 
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader

# ==========================================
# 1. HÀM HUẤN LUYỆN GIAI ĐOẠN 1
# ==========================================
def train_stage_1(train_loader, test_loader):
    print("\n" + "="*50)
    print(" BẮT ĐẦU GIAI ĐOẠN 1: Train ECG AutoEncoder")
    print("="*50)
    
    model = Stage1_ECG_AutoEncoder(dims=[64, 128, 256]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR_STAGE_1, weight_decay=1e-4)
    criterion = nn.MSELoss() 
    
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS_STAGE_1):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Stage 1 - Epoch {epoch+1}/{EPOCHS_STAGE_1} [Train]")
        
        for ecg, ppg, _ in loop:
            ecg = ecg.float().to(device) 
            if ecg.dim() == 2:
                ecg = ecg.unsqueeze(1)
            optimizer.zero_grad()
            ecg_hat = model(ecg)
            loss = criterion(ecg_hat, ecg)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for ecg, ppg, _ in test_loader:
                ecg = ecg.float().to(device)
                if ecg.dim() == 2:
                    ecg = ecg.unsqueeze(1)
                ecg_hat = model(ecg)
                loss = criterion(ecg_hat, ecg)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(test_loader)
        print(f"-> Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "saved_models/best_stage1_ecg_ae.pth")
            print("   [+] Đã lưu mô hình Stage 1 tốt nhất!")

    print("Hoàn thành Giai đoạn 1!\n")
    return model

# ==========================================
# 2. HÀM HUẤN LUYỆN GIAI ĐOẠN 2 (IMPROVED)
# ==========================================
def train_stage_2(train_loader, test_loader):
    print("\n" + "="*50)
    print(" STARTING STAGE 2: Train PPG2ECG (Improved)")
    print("="*50)
    
    # Tải mô hình Stage 1 (ECG Teacher)
    pretrained_stage1 = Stage1_ECG_AutoEncoder(dims=[64, 128, 256]).to(device)
    pretrained_stage1.load_state_dict(torch.load("saved_models/best_stage1_ecg_ae.pth"))
    pretrained_stage1.eval()
    
    # Khởi tạo mô hình Stage 2 MỚI
    # use_derivatives có thể bật/tắt tùy ý, mặc định = False
    model = Stage2_PPG2ECG_Improved(pretrained_ecg_model=pretrained_stage1, dims=[64, 128, 256], use_derivatives=False).to(device)
    
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=LR_STAGE_2, weight_decay=1e-4)
    
    # Hàm Loss MỚI
    criterion = Stage2LossImproved(lambda_l1=1.0, lambda_mse=0.5, lambda_pearson=1.0, lambda_latent=0.1).to(device)
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS_STAGE_2):
        model.train()
        train_loss = 0.0
        metrics_tot = {"l1": 0.0, "mse": 0.0, "pearson": 0.0, "latent": 0.0}
        
        loop = tqdm(train_loader, desc=f"Stage 2 - Epoch {epoch+1}/{EPOCHS_STAGE_2} [Train]")
        
        for ecg, ppg, _ in loop:
            ppg = ppg.float().to(device)
            ecg = ecg.float().to(device)
            if ppg.dim() == 2: ppg = ppg.unsqueeze(1)
            if ecg.dim() == 2: ecg = ecg.unsqueeze(1)
            
            optimizer.zero_grad()
            
            # Forward pass MỚI: Truyền ecg_real để trích xuất latent_ecg_teacher
            out = model(ppg, ecg_real=ecg)
            
            # Tính loss: criterion trả về (total_loss, metrics_dict)
            loss, metrics = criterion(out, ecg)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            for k in metrics_tot.keys():
                metrics_tot[k] += metrics[k].item()
            
            loop.set_postfix(Tot=loss.item(), MSE=metrics["mse"].item(), Pear=metrics["pearson"].item(), Lat=metrics["latent"].item())
            
        num_train = len(train_loader)
        avg_train_loss = train_loss / num_train
        
        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        val_metrics_tot = {"l1": 0.0, "mse": 0.0, "pearson": 0.0, "latent": 0.0}
        
        with torch.no_grad():
            for ecg, ppg, _ in test_loader:
                ppg = ppg.float().to(device)
                ecg = ecg.float().to(device)
                if ppg.dim() == 2: ppg = ppg.unsqueeze(1)
                if ecg.dim() == 2: ecg = ecg.unsqueeze(1)
                
                # Val cũng truyền ecg_real để theo dõi latent loss
                out = model(ppg, ecg_real=ecg)
                v_loss, v_metrics = criterion(out, ecg)
                
                val_loss += v_loss.item()
                for k in val_metrics_tot.keys():
                    val_metrics_tot[k] += v_metrics[k].item()
                
        num_val = len(test_loader)
        avg_val_loss = val_loss / num_val
        
        print(f"-> Epoch {epoch+1}:")
        print(f"   Train: Tot={avg_train_loss:.4f} | MSE={metrics_tot['mse']/num_train:.4f} | Pear={metrics_tot['pearson']/num_train:.4f} | Latent={metrics_tot['latent']/num_train:.4f}")
        print(f"   Val  : Tot={avg_val_loss:.4f} | MSE={val_metrics_tot['mse']/num_val:.4f} | Pear={val_metrics_tot['pearson']/num_val:.4f} | Latent={val_metrics_tot['latent']/num_val:.4f}")
        
        if avg_train_loss < best_val_loss:
            best_val_loss = avg_train_loss
            torch.save(model.state_dict(), "saved_models/best_stage2_ppg2ecg.pth")
            print("   [+] Saved best Stage 2 model!")

# ==========================================
# 3. HÀM CHÍNH (MAIN)
# ==========================================
if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders()
    
    # Chạy Giai đoạn 1 (Có thể comment lại nếu bạn đã train xong và chỉ muốn chạy tiếp GĐ 2)
    # train_stage_1(train_loader, test_loader)
    
    # Chạy Giai đoạn 2
    train_stage_2(train_loader, test_loader)