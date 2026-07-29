import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import random
import os
import types  # Dùng để ép kiểu hàm monkey-patch

from load_data import LoadData
from ecg2ecg import ECGAutoencoder, ECGAEConfig 
from metric import calculate_cosine_similarity, calculate_dtw_distance, calculate_metrics
from scipy.signal import correlate

# ==========================================
# --- CONFIGURATION ---
# ==========================================
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
SEQ_LENGTH = 2400

# Đường dẫn data test và trọng số model
TEST_DATA_PATH = '/home/linhhima/Diffusion_datasets/combined_segment_split_test.npz'
MODEL_PATH = '/home/linhhima/PPG_ECG/saved_models_ecg_vae_segment/best_ecg_autoencoder.pth'


# ============================================================
# --- MONKEY-PATCH FUNCTIONS (Hàm vá cấu trúc nội bộ) ---
# ============================================================
def patched_decoder_forward(self, z: torch.Tensor) -> dict:
    """Hàm forward vá lỗi cho ECGDecoder1D để trích xuất toàn bộ biến trung gian."""
    x = self.in_proj(z)
    x = self.decoder_bottleneck_attn(x)

    x = self.stage4(x)
    x = self.up3(x)
    x = self.stage3(x)
    x = self.up2(x)
    x = self.stage2(x)
    x = self.up1(x)
    x = self.stage1(x)

    local_out = self.local_head(x)   # [B, 1, 2400]

    z_flat = z.reshape(z.size(0), -1)
    z_global = self.to_global(z_flat)
  
    trend_out = self.trend_layer(z_global)
    level_out = self.level_layer(z_global)
    global_out = level_out + trend_out  # [B, 1, 2400]

    out = local_out + global_out
    out = self.out_refine(out)
    
    return {
        "reconstructed_ecg": out,
        "local_out": local_out,
        "global_out": global_out,
        "trend": trend_out,
        "level": level_out
    }

def patched_autoencoder_forward(self, ecg: torch.Tensor) -> dict:
    """Hàm forward vá lỗi cho lớp tổng ECGAutoencoder."""
    feat = self.encoder(ecg)                       # [B, 512, 75]
    z = self.latent_head(feat)                     # [B, latent_channels, 75]
    
    decoder_outputs = self.decoder(z)
    decoder_outputs["latent_ecg"] = z
    
    return decoder_outputs


# ==========================================
# --- MAIN TEST LOGIC ---
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_model():
    print(f"[*] Đang tải mô hình và cấu hình mạng nâng cao lên {DEVICE}...")
    
    cfg = ECGAEConfig(
        input_length=SEQ_LENGTH,
        in_channels=1,
        dims=(64, 128, 256, 512),
        depths=(2, 2, 4, 2),
        latent_channels=16,
        latent_length=75,
        attn_heads=8,
        attn_dropout=0.0,
        global_latent_dim=128,
        trend_poly=2,
    )
    
    model = ECGAutoencoder(cfg).to(DEVICE)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"[!] Không tìm thấy file trọng số tại: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.decoder.forward = types.MethodType(patched_decoder_forward, model.decoder)
    model.forward = types.MethodType(patched_autoencoder_forward, model)

    model.eval()
    return model

def visualize_results(ecg_true, ecg_pred, local_out, global_out, trend, level, record_name):
    print(f"[*] Trực quan hóa thành phần mạng ECGDecoder1D: {record_name}")
    t_ecg = np.arange(len(ecg_true))

    # Sử dụng subplots để ép trục X (sharex=True) thẳng hàng hoàn hảo từ trên xuống dưới
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    # plt.suptitle(f"ECGDecoder1D Component Disentanglement | Record: {record_name}", fontsize=14, fontweight='bold')

    # Subplot 1: Tổng thể
    axes[0].plot(t_ecg, ecg_true, color='black', label='Ground Truth ECG', alpha=0.7, linewidth=1.5)
    axes[0].plot(t_ecg, ecg_pred, color='red', label='Final Output (after out_refine)', linestyle='--', alpha=0.85, linewidth=1.2)
    axes[0].set_title("1. Overall ECG Reconstruction", fontsize=11, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right')

    # Subplot 2: Nhánh 1 - local_out
    axes[1].plot(t_ecg, local_out, color='teal', label='First Branch: local_out (Morphology & Peaks)', linewidth=1.2)
    axes[1].axhline(0, color='gray', linestyle=':', alpha=0.5)
    axes[1].set_title("2. First Branch (Local Details - Zero Centered)", fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')

    # Subplot 3: Nhánh 2 - global_out tổng thể
    axes[2].plot(t_ecg, global_out, color='darkorange', label='Second Branch: global_out (Combined Baseline)', linewidth=2.5)
    axes[2].set_title("3. Second Branch (Global Component: Level + Trend)", fontsize=11, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='upper right')

    # Subplot 4: BÓC TÁCH NỘI BỘ NHÁNH 2 (Đường nét được làm đậm và rõ ràng hơn)
    # Tăng cường độ dày và opacity cho tổng thể đường màu cam
    axes[3].plot(t_ecg, global_out, color='darkorange', label='Total global_out Baseline Platform', alpha=0.7, linewidth=3.5)
    # Đổi Trend sang nét đứt dày màu đỏ thẫm (crimson) cực rõ
    axes[3].plot(t_ecg, trend, color='crimson', label='trend_layer [Polynomial Curve]', linestyle='--', linewidth=2.5)
    # Đổi Level sang nét chấm gạch dày màu xanh đại dương đậm (navy) để tăng tương phản
    axes[3].plot(t_ecg, level, color='navy', label='level_layer [Constant Baseline Offset]', linestyle='-.', linewidth=2.5)
    
    axes[3].set_title("4. Decomposition of Second Branch (Trend vs. Level Layer)", fontsize=11, fontweight='bold')
    axes[3].set_xlabel("Time Samples", fontsize=10)
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc='upper right')

    plt.tight_layout()
    plt.show()

def run_visualization():
    set_seed(SEED)
    model = load_model()
    seen_records = set()

    try:
        test_dataset = LoadData(TEST_DATA_PATH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    except Exception as e:
        print(f"[!] Lỗi khi load data: {e}")
        return
   
    print("\n[*] Đang chạy suy luận phân tách luồng đặc trưng...")
    with torch.no_grad():
        for batch_data in test_loader:
            ecg = batch_data[0]
            record_names = batch_data[2] 

            ecg_input = ecg.float().unsqueeze(1).to(DEVICE) if ecg.dim() == 2 else ecg.float().to(DEVICE)

            outputs = model(ecg_input)
            
            predicted_ecg = outputs["reconstructed_ecg"]
            local_out = outputs["local_out"]
            global_out = outputs["global_out"]
            trend = outputs["trend"]
            level = outputs["level"]

            ecg_true_np = ecg_input.cpu().squeeze(1).numpy()
            ecg_pred_np = predicted_ecg.cpu().squeeze(1).numpy()
            local_np = local_out.cpu().squeeze(1).numpy()
            global_np = global_out.cpu().squeeze(1).numpy()
            trend_np = trend.cpu().squeeze(1).numpy()
            level_np = level.cpu().squeeze(1).numpy()

            for idx in range(ecg_true_np.shape[0]):
                current_rec_name = record_names[idx]
                
                if current_rec_name not in seen_records:
                    visualize_results(
                        ecg_true=ecg_true_np[idx], 
                        ecg_pred=ecg_pred_np[idx], 
                        local_out=local_np[idx],
                        global_out=global_np[idx],
                        trend=trend_np[idx],
                        level=level_np[idx],
                        record_name=current_rec_name
                    )
                    seen_records.add(current_rec_name)
                
                if len(seen_records) >= 5:
                    print("[*] Đã hiển thị đủ 5 samples trực quan cấu trúc.")
                    return

if __name__ == "__main__":
    run_visualization()