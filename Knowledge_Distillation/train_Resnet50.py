from torch import nn
import torch
import torch.nn.functional as F
import random
import numpy as np
from load_data import LoadData
from torch.utils.data import DataLoader
from loss import PearsonCorrelationLoss
from ResNet50 import ResNet50_1D
from CLIP import DualDomainEncoder, ECGDecoder_Transformer

OUTPUT_EMBED_DIM = 128  
LAYERS = [3, 4, 6, 3] 
BASE_WIDTH = 64
EXPANSION = 4 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(29)
    NUM_EPOCHS = 200
    best_loss = float('inf')
    save_path = "ppg_ecg.pth"
    # encoder_path = "best_autoencoder_ecg.pth"
    
    train_dataset = LoadData("/home/linhhima/PPG_ECG/processed_data/mimic3_v1_2400_train.npz")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
   
    # encoder = ResNet50_1D().to(device)
    encoder = DualDomainEncoder(fft_dim=256, ecg_branch=False).to(device)
    # print(f"Loading pre-trained PPG encoder from {encoder_path}...")
    # try:
    #     checkpoint_enc = torch.load(encoder_path, map_location=device)
    #     encoder.load_state_dict(checkpoint_enc['encoder_state_dict'])
    #     print("Successfully loaded encoder weights! Both Encoder and Decoder will be fine-tuned.")
    # except Exception as e:
    #     print(f"Lỗi khi load encoder: {e}. Vui lòng kiểm tra lại file {encoder_path}")
    #     exit()
    
    decoder = ECGDecoder_Transformer(
            bottleneck_channels=2048, 
            d_model=256, 
            nhead=8, 
            num_layers=4, 
            target_length=2400
        ).to(device)
    
    mse_loss = torch.nn.MSELoss().to(device)
    pearson_loss = PearsonCorrelationLoss().to(device)
    
    # --- THAY ĐỔI 1: TỐI ƯU HÓA CẢ HAI MÔ HÌNH ---
    # Kết hợp các tham số của cả encoder và decoder vào cùng một Optimizer
    params_to_optimize = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params_to_optimize, lr=1e-4)

    for epoch in range(1, NUM_EPOCHS + 1):
        # --- THAY ĐỔI 2: CHUYỂN CẢ 2 MÔ HÌNH SANG CHẾ ĐỘ TRAIN ---
        encoder.train()
        decoder.train()
        
        total_loss = 0.0 
        total_mse_loss = 0.0
        total_pearson_loss = 0.0
        
        for i, (ecg, ppg, _) in enumerate(train_loader):
            ppg = ppg.to(device).float().unsqueeze(1)
            ecg = ecg.to(device).float().unsqueeze(1)

            # --- THAY ĐỔI 3: BỎ torch.no_grad() ĐỂ TÍNH ĐẠO HÀM CHO ENCODER ---
            # Trước đây có with torch.no_grad(): ở đây, giờ ta bỏ đi.
            ppg_fused_1d, ppg_fused_3d, ppg_features_list = encoder(ppg)
                
            recon_ecg = decoder(ppg_fused_3d)
            loss_mse = mse_loss(recon_ecg, ecg)
            loss_pearson = pearson_loss(recon_ecg, ecg)
            loss = loss_mse + loss_pearson
            loss =  loss_pearson

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_mse_loss += loss_mse.item()
            total_pearson_loss += loss_pearson.item()
            
        num_batches = len(train_loader)
        avg_loss = total_loss / num_batches
        avg_mse = total_mse_loss / num_batches
        avg_pearson = total_pearson_loss / num_batches
        
        print(f"Epoch [{epoch}/{NUM_EPOCHS}] - Total Loss: {avg_loss:.4f} ")

        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"   => Found new best model with Total Loss: {best_loss:.4f}. Saving...")
            
            checkpoint = {
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss
            }
            torch.save(checkpoint, save_path)
            
    print(f"Training completed. Best loss achieved: {best_loss:.4f}")