import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt

from load_data import LoadData
from model import Stage1_ECG_AutoEncoder, Stage2_PPG2ECG_Improved

# ==========================================
# 0. HYPERPARAMETERS & DEVICE
# ==========================================
BATCH_SIZE = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

TEST_DATA_PATH = '/home/linhhima/Diffusion datasets/test.npz' 

def get_test_dataloader():
    test_loader = DataLoader(
        LoadData(TEST_DATA_PATH), 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    return test_loader

# ==========================================
# 1. UTILS
# ==========================================
def calculate_pearson(pred, target):
    pred_mean = pred.mean(dim=-1, keepdim=True)
    target_mean = target.mean(dim=-1, keepdim=True)
    pred_centered = pred - pred_mean
    target_centered = target - target_mean
    cov = (pred_centered * target_centered).sum(dim=-1)
    var_pred = (pred_centered ** 2).sum(dim=-1)
    var_target = (target_centered ** 2).sum(dim=-1)
    pearson = cov / (torch.sqrt(var_pred * var_target) + 1e-8)
    return pearson.mean().item()

def visualize_and_save(batch_idx, ecg_real, ecg_hat, ppg=None, stage_name="Stage2", num_samples=3):
    save_dir = f"results/visualizations/{stage_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    ecg_real_np = ecg_real.cpu().detach().numpy().squeeze(1)
    ecg_hat_np = ecg_hat.cpu().detach().numpy().squeeze(1)
    ppg_np = ppg.cpu().detach().numpy().squeeze(1) if ppg is not None else None
    
    n = min(num_samples, ecg_real_np.shape[0])
    
    for i in range(n):
        plt.figure(figsize=(12, 6 if ppg is not None else 4))
        
        if ppg is not None:
            plt.subplot(2, 1, 1)
            plt.plot(ppg_np[i], color='green', label='Input PPG')
            plt.title(f'Batch {batch_idx} - Sample {i+1} | Input PPG')
            plt.legend(loc="upper right")
            plt.grid(True, linestyle='--', alpha=0.6)
            
            plt.subplot(2, 1, 2)
            plt.plot(ecg_real_np[i], color='blue', label='Ground Truth ECG', alpha=0.7)
            plt.plot(ecg_hat_np[i], color='red', linestyle='--', label='Predicted ECG', alpha=0.9)
            plt.title('Output: Real vs Predicted ECG')
            plt.legend(loc="upper right")
            plt.grid(True, linestyle='--', alpha=0.6)
        else:
            plt.plot(ecg_real_np[i], color='blue', label='Original ECG', alpha=0.7)
            plt.plot(ecg_hat_np[i], color='red', linestyle='--', label='Reconstructed ECG', alpha=0.9)
            plt.title(f'Batch {batch_idx} - Sample {i+1} | AutoEncoder Reconstruction')
            plt.legend(loc="upper right")
            plt.grid(True, linestyle='--', alpha=0.6)
            
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"batch_{batch_idx}_sample_{i+1}.png"), dpi=200)
        plt.close()

# ==========================================
# 2. STAGE 1 TEST
# ==========================================
def test_stage_1(test_loader):
    print("\n" + "="*50)
    print(" EVALUATING STAGE 1: ECG AutoEncoder")
    print("="*50)
    
    model = Stage1_ECG_AutoEncoder(dims=[64, 128, 256]).to(device)
    model_path = "/home/linhhima/PPG_ECG/saved_models/best_stage1_ecg_ae.pth"
    
    if not os.path.exists(model_path):
        print(f"⚠️ File not found at {model_path}.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() 
    
    mse_criterion = nn.MSELoss()
    l1_criterion = nn.L1Loss()
    total_mse, total_mae, total_pearson = 0.0, 0.0, 0.0
    
    with torch.no_grad():
        loop = tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing Stage 1")
        for batch_idx, (ecg, ppg, _) in loop:
            ecg = ecg.float().to(device)
            ppg = ppg.float().to(device)

            if ecg.dim() == 2: ecg = ecg.unsqueeze(1)
            if ppg.dim() == 2: ppg = ppg.unsqueeze(1)

            ecg_hat = model(ppg)
            
            total_mse += mse_criterion(ecg_hat, ecg).item()
            total_mae += l1_criterion(ecg_hat, ecg).item()
            total_pearson += calculate_pearson(ecg_hat, ecg)
            
            if batch_idx < 2:
                visualize_and_save(batch_idx, ecg, ecg_hat, ppg=None, stage_name="Stage1_AutoEncoder")
            
    num_batches = len(test_loader)
    print(f"\n✅ STAGE 1 RESULTS:")
    print(f"   - Reconstruction MSE     : {total_mse / num_batches:.4f}")
    print(f"   - Reconstruction MAE     : {total_mae / num_batches:.4f}")
    print(f"   - Reconstruction Pearson : {total_pearson / num_batches:.4f}")

# ==========================================
# 3. STAGE 2 TEST (IMPROVED)
# ==========================================
def test_stage_2(test_loader):
    print("\n" + "="*50)
    print(" EVALUATING STAGE 2: PPG to ECG Generator")
    print("="*50)
    
    pretrained_stage1 = Stage1_ECG_AutoEncoder(dims=[64, 128, 256]).to(device)
    stage1_path = "/home/linhhima/PPG_ECG/saved_models/best_stage1_ecg_ae.pth"
    if os.path.exists(stage1_path):
        pretrained_stage1.load_state_dict(torch.load(stage1_path, map_location=device))
    pretrained_stage1.eval()

    model = Stage2_PPG2ECG_Improved(pretrained_ecg_model=pretrained_stage1, dims=[64, 128, 256], use_derivatives=False).to(device)
    model_path = "/home/linhhima/PPG_ECG/saved_models/best_stage2_ppg2ecg.pth"
    
    if not os.path.exists(model_path):
        print(f"⚠️ File not found at {model_path}.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() 
    
    mse_criterion = nn.MSELoss()
    l1_criterion = nn.L1Loss()
    total_mse, total_mae, total_pearson = 0.0, 0.0, 0.0
    
    with torch.no_grad():
        loop = tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing Stage 2")
        for batch_idx, (ecg, ppg, _) in loop:
            ppg = ppg.float().to(device)
            ecg = ecg.float().to(device)
            
            if ppg.dim() == 2: ppg = ppg.unsqueeze(1)
            if ecg.dim() == 2: ecg = ecg.unsqueeze(1)
            
            out = model(ppg)
            ecg_hat = out["ecg_hat"]
            
            total_mse += mse_criterion(ecg_hat, ecg).item()
            total_mae += l1_criterion(ecg_hat, ecg).item()
            total_pearson += calculate_pearson(ecg_hat, ecg)
            
            if batch_idx < 2:
                visualize_and_save(batch_idx, ecg, ecg_hat, ppg=ppg, stage_name="Stage2_PPG2ECG")
            
    num_batches = len(test_loader)
    print(f"\n✅ STAGE 2 RESULTS (PPG -> ECG):")
    print(f"   - Generation MSE     : {total_mse / num_batches:.4f}")
    print(f"   - Generation MAE     : {total_mae / num_batches:.4f}")
    print(f"   - Generation Pearson : {total_pearson / num_batches:.4f}")

# ==========================================
# 4. MAIN
# ==========================================
if __name__ == "__main__":
    test_loader = get_test_dataloader()
    test_stage_1(test_loader)
    # test_stage_2(test_loader)