import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import pearsonr
from scipy.signal import correlate

# Import các module kiến trúc của Linh
from load_data import LoadData
from ResNet50 import ResNet50_1D
from CLIP import DualDomainEncoder, ECGDecoder_Transformer

# ==========================================
# 1. CÔNG CỤ CĂN CHỈNH TÍN HIỆU (SHIFT ALIGNMENT)
# ==========================================
def align_signals(target, pred, max_lag=60):
    """
    Tìm độ trễ tối ưu (Pulse Transit Time) và dịch chuyển pred để khớp với target.
    """
    t_norm = target - np.mean(target)
    p_norm = pred - np.mean(pred)
    
    cross_corr = correlate(t_norm, p_norm, mode='full')
    lags = np.arange(-len(pred) + 1, len(target))
    
    mask = (lags >= -max_lag) & (lags <= max_lag)
    valid_lags = lags[mask]
    valid_corr = cross_corr[mask]
    
    best_lag = valid_lags[np.argmax(valid_corr)]
    
    if best_lag > 0:
        shifted_pred = np.pad(pred, (best_lag, 0), mode='edge')[:len(pred)]
    elif best_lag < 0:
        shifted_pred = np.pad(pred, (0, abs(best_lag)), mode='edge')[abs(best_lag):]
    else:
        shifted_pred = pred.copy()
        
    return shifted_pred, best_lag

def set_seed(seed=29):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ==========================================
# 2. CHƯƠNG TRÌNH TEST CHÍNH
# ==========================================
def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(29)
    
    # Đường dẫn file
    save_path = "/home/linhhima/PPG_ECG/draft_branch_27/ppg_ecg.pth"
    test_data_path = "/home/linhhima/PPG_ECG/processed_data/mimic3_v1_2400_test.npz"

    print(f"[*] Loading data from: {test_data_path}")
    test_dataset = LoadData(test_data_path)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Khởi tạo Model
    print(f"[*] Initializing models on {device}...")
    # encoder = ResNet50_1D().to(device)
    encoder = DualDomainEncoder(fft_dim=256, ecg_branch=False).to(device)
    decoder = ECGDecoder_Transformer(
        bottleneck_channels=2048, 
        d_model=256, 
        nhead=8, 
        num_layers=4, 
        target_length=2400
    ).to(device)

    # Load Checkpoint
    if os.path.exists(save_path):
        checkpoint = torch.load(save_path, map_location=device)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        print(f"[+] Loaded model from epoch {checkpoint.get('epoch', 'N/A')}")
    else:
        print(f"[!] Error: {save_path} not found!"); return

    encoder.eval()
    decoder.eval()

    # Containers lưu kết quả toàn dataset
    metrics = {
        "raw_rmse": [], "raw_pearson": [],
        "aligned_rmse": [], "aligned_pearson": [],
        "lags": []
    }
    
    # Chỉ lưu 1 mẫu duy nhất để vẽ hình
    viz_sample = None

    print("[*] Starting inference and alignment...")
    with torch.no_grad():
        for i, (ecg, ppg, _) in enumerate(tqdm(test_loader, desc="Testing")):
            ppg = ppg.to(device).float().unsqueeze(1)
            ecg = ecg.to(device).float().unsqueeze(1)

            ppg_fused_1d, ppg_fused_3d, ppg_features_list= encoder(ppg)
            recon_ecg = decoder(ppg_fused_3d)

            # Chuyển sang numpy để xử lý từng mẫu
            targets = ecg.squeeze(1).cpu().numpy()
            preds = recon_ecg.squeeze(1).cpu().numpy()

            for b in range(targets.shape[0]):
                y_true = targets[b]
                y_pred = preds[b]

                # 1. Metric gốc (Raw)
                r_rmse = np.sqrt(np.mean((y_true - y_pred)**2))
                r_p, _ = pearsonr(y_true, y_pred)
                
                # 2. Metric sau khi Align
                y_shifted, lag = align_signals(y_true, y_pred, max_lag=60)
                s_rmse = np.sqrt(np.mean((y_true - y_shifted)**2))
                s_p, _ = pearsonr(y_true, y_shifted)

                # Lưu trữ kết quả vào list tổng
                metrics["raw_rmse"].append(r_rmse)
                metrics["raw_pearson"].append(r_p)
                metrics["aligned_rmse"].append(s_rmse)
                metrics["aligned_pearson"].append(s_p)
                metrics["lags"].append(lag)

                # Lưu lại mẫu đầu tiên để vẽ đồ thị
                if viz_sample is None:
                    viz_sample = {
                        'true': y_true, 'pred': y_pred, 
                        'shifted': y_shifted, 'lag': lag, 'pearson': s_p
                    }

    # 3. HIỂN THỊ THỐNG KÊ TỔNG QUAN
    avg_results = {k: np.mean(v) for k, v in metrics.items()}
    
    print("\n" + "="*65)
    print(f"{'DATASET EVALUATION SUMMARY (N=' + str(len(metrics['lags'])) + ')':^65}")
    print("="*65)
    print(f"{'Metric':<25} | {'Raw (Before)':<15} | {'Aligned (After)':<15}")
    print("-" * 65)
    print(f"{'Pearson Correlation ↑':<25} | {avg_results['raw_pearson']:<15.4f} | {avg_results['aligned_pearson']:<15.4f}")
    print(f"{'RMSE ↓':<25} | {avg_results['raw_rmse']:<15.4f} | {avg_results['aligned_rmse']:<15.4f}")
    print("-" * 65)
    print(f"Average absolute phase lag: {np.mean(np.abs(metrics['lags'])):.2f} samples")
    print(f"Pearson improvement: {((avg_results['aligned_pearson'] - avg_results['raw_pearson'])/avg_results['raw_pearson']*100):.2f}%")
    print("="*65)

    # 4. VẼ 1 MẪU TIÊU BIỂU (VISUALIZATION)
    if viz_sample:
        print("\n[*] Plotting 1 sample reconstruction...")
        plt.figure(figsize=(15, 6))
        
        # Zoom vào đoạn giữa (samples 400 đến 1200) để thấy rõ chi tiết sóng
        start, end = 400, 1200 
        time_idx = np.arange(start, end)

        plt.plot(time_idx, viz_sample['true'][start:end], color='black', alpha=0.3, lw=4, label='ECG Ground Truth')
        plt.plot(time_idx, viz_sample['pred'][start:end], color='blue', ls='--', alpha=0.6, label='Raw Prediction')
        plt.plot(time_idx, viz_sample['shifted'][start:end], color='red', lw=1.5, label=f'Aligned Prediction (Lag: {viz_sample["lag"]})')
        
        plt.title(f"ECG Signal Reconstruction - Aligned Pearson: {viz_sample['pearson']:.4f}", fontsize=14, fontweight='bold')
        plt.xlabel("Samples", fontsize=12)
        plt.ylabel("Amplitude (Z-score)", fontsize=12)
        plt.legend(loc='upper right')
        plt.grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    run_evaluation()