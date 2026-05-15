import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import os

from scipy.signal import correlate
from scipy.signal import butter, filtfilt, find_peaks
from load_data import LoadData 
from ecg2ecg import ECGAutoencoder, ECGAEConfig
from ppg2ecg import PPG2ECGModel, PPG2ECGConfig 
from flow_model import LatentRectifiedFlow 

# Giả sử bạn có file metric chứa các hàm này
from metric import calculate_cosine_similarity, calculate_dtw_distance, calculate_metrics

# ==========================================
# CẤU HÌNH (CONFIGURATIONS)
# ==========================================
SEED = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
INPUT_LENGTH = 2400
TEST_DATA_PATH = '/home/linhhima/PPG_ECG/datasets/z_score_norm/total_mimic_af.npz'

# Đường dẫn trọng số (Hãy đảm bảo đường dẫn chính xác)
PHASE1_ECG_PATH = '/home/linhhima/PPG_ECG/Result_model2_subjects/saved_models_ecg_vae/best_ecg_autoencoder.pth'
PHASE1_PPG_PATH = '/home/linhhima/PPG_ECG/Result_model2_subjects/saved_models_alignment/best_ppg_alignment.pth'
PHASE2_FLOW_PATH = '/home/linhhima/PPG_ECG/Result_model2_subjects/saved_models_flow/best_rectified_flow.pth'

ODE_STEPS = 10 # Số bước giải Euler cho Rectified Flow

# ==========================================
# CÁC HÀM TIỆN ÍCH & LOAD MÔ HÌNH
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def calculate_rr_intervals(r_peaks: np.ndarray, fs: int = 125):
    """
    Tính khoảng cách R-R trung bình, Nhịp tim (BPM) và SDNN.
    
    Tham số:
    - r_peaks: Mảng numpy chứa vị trí (index) của các đỉnh R.
    - fs: Tần số lấy mẫu (Sampling rate), mặc định là 125 Hz.
    
    Trả về:
    - avg_rr_ms: Khoảng cách R-R trung bình (mili-giây).
    - hr_bpm: Nhịp tim trung bình (Nhịp/phút).
    - sdnn_ms: Độ lệch chuẩn của các khoảng R-R (mili-giây) - Đại diện cho biến thiên nhịp tim.
    """
    # Cần ít nhất 2 đỉnh để tạo thành 1 khoảng R-R
    if len(r_peaks) < 2:
        return 0.0, 0.0, 0.0
    
    # Tính khoảng cách giữa các đỉnh liên tiếp (đơn vị: số lượng mẫu)
    rr_intervals_samples = np.diff(r_peaks)
    
    # Chuyển đổi từ số lượng mẫu sang mili-giây (ms)
    # Công thức: (số mẫu / tần số lấy mẫu) * 1000
    rr_intervals_ms = (rr_intervals_samples / fs) * 1000.0
    
    # 1. Trung bình khoảng cách R-R
    avg_rr_ms = np.mean(rr_intervals_ms)
    
    # 2. Nhịp tim trung bình (BPM: Beats Per Minute)
    # 60,000 mili-giây = 1 phút
    hr_bpm = 60000.0 / avg_rr_ms if avg_rr_ms > 0 else 0.0
    
    # 3. SDNN (Standard Deviation of NN intervals) - Thước đo HRV
    sdnn_ms = np.std(rr_intervals_ms)
    
    return avg_rr_ms, hr_bpm, sdnn_ms

def pan_tompkins_qrs(ecg_signal: np.ndarray, fs: int = 125, external_threshold: float = None, return_threshold: bool = False):
    """
    Thuật toán Pan-Tompkins đã chỉnh sửa để nhận/trả ngưỡng.
    """
    ecg_signal = np.array(ecg_signal).flatten()
    
    # 1. Bộ lọc dải thông (Bandpass Filter: 5 - 15 Hz)
    nyq = 0.5 * fs
    low = 5.0 / nyq
    high = 15.0 / nyq
    b, a = butter(1, [low, high], btype='band')
    filtered_ecg = filtfilt(b, a, ecg_signal)
    
    # 2. Đạo hàm (Derivative)
    diff_ecg = np.diff(filtered_ecg)
    diff_ecg = np.insert(diff_ecg, 0, diff_ecg[0])
    
    # 3. Bình phương (Squaring function)
    squared_ecg = diff_ecg ** 2
    
    # 4. Tích phân cửa sổ trượt (Moving Window Integration)
    window_width = int(0.15 * fs)
    integrated_ecg = np.convolve(squared_ecg, np.ones(window_width) / window_width, mode='same')
    
    # 5. THRESHOLDING (Khúc này đã được sửa)
    # Nếu có truyền ngưỡng từ ngoài vào thì dùng nó, nếu không thì tự tính trung bình
    if external_threshold is not None:
        threshold = external_threshold
    else:
        threshold = np.mean(integrated_ecg)
        
    min_distance = int(0.3 * fs)
    peaks_integrated, _ = find_peaks(integrated_ecg, height=threshold, distance=min_distance)
    
    # 6. Đối chiếu lại để tìm chính xác đỉnh R trên tín hiệu gốc (Back-search)
    r_peaks = []
    search_window = int(0.05 * fs) 
    
    for p in peaks_integrated:
        start = max(0, p - search_window)
        end = min(len(ecg_signal), p + search_window)
        if start < end:
            local_max = np.argmax(ecg_signal[start:end])
            r_peaks.append(start + local_max)
            
    # Trả về cả đỉnh R và cái ngưỡng đã dùng (nếu được yêu cầu)
    if return_threshold:
        return np.array(r_peaks), threshold
    
    return np.array(r_peaks)

def calculate_peak_count_ratio(true_s: np.ndarray, pred_s: np.ndarray, sampling_rate=125):
    """
    Đếm số lượng đỉnh R bằng thuật toán Pan-Tompkins và tính tỷ lệ.
    ĐÃ SỬA: Ép Predicted ECG phải dùng ngưỡng tính được từ Ground Truth.
    """
    # 1. Tìm đỉnh R của Ground Truth và TRÍCH XUẤT NGƯỠNG (gt_threshold)
    true_peaks, gt_threshold = pan_tompkins_qrs(true_s, fs=sampling_rate, return_threshold=True)
    
    # 2. Truyền ngưỡng của Ground Truth cho Predicted ECG
    pred_peaks = pan_tompkins_qrs(pred_s, fs=sampling_rate, external_threshold=gt_threshold, return_threshold=False)
    
    # 3. Đếm tổng số đỉnh
    num_true = len(true_peaks)
    num_pred = len(pred_peaks)

    # true_avg_rr, true_hr, true_sdnn = calculate_rr_intervals(true_peaks, fs=125)
    # pred_avg_rr, pred_hr, pred_sdnn = calculate_rr_intervals(pred_peaks, fs=125)

    # # In ra để kiểm tra ngay lập tức cho Record hiện tại
    # print(f"  GT   -> RR: {true_avg_rr:.1f} ms | HR: {true_hr:.1f} bpm | SDNN: {true_sdnn:.1f} ms")
    # print(f"  Pred -> RR: {pred_avg_rr:.1f} ms | HR: {pred_hr:.1f} bpm | SDNN: {pred_sdnn:.1f} ms")
    # print("-" * 50)
    
    # 4. Tính tỷ lệ dự đoán so với thực tế (Prediction Ratio)
    if num_true > 0:
        ratio = num_pred / num_true
    else:
        ratio = 1.0 if num_pred == 0 else 0.0
        
    return num_true, num_pred, ratio

def load_models():
    print(f"[*] Loading Models on {DEVICE}...")
    
    # 1. Load ECG Autoencoder (Teacher & Decoder)
    ecg_cfg = ECGAEConfig(
        input_length=INPUT_LENGTH, in_channels=1, dims=(64, 128, 256, 512),
        depths=(2, 2, 4, 2), latent_channels=16, latent_length=75,
        attn_heads=8, attn_dropout=0.0, global_latent_dim=128, trend_poly=2
    )
    ecg_ae = ECGAutoencoder(ecg_cfg).to(DEVICE)
    ckpt_ecg = torch.load(PHASE1_ECG_PATH, map_location=DEVICE)
    ecg_ae.load_state_dict(ckpt_ecg.get("model_state_dict", ckpt_ecg))
    ecg_ae.eval()
    for p in ecg_ae.parameters(): p.requires_grad = False

    # 2. Load PPG2ECG Alignment Model (Lấy z_ppg)
    ppg_cfg = PPG2ECGConfig(
        input_length=INPUT_LENGTH, ppg_in_channels=1, dims=(64, 128, 256, 512),
        depths=(2, 2, 4, 2), latent_channels=16, latent_length=75,
        attn_heads=8, attn_dropout=0.0, use_derivatives=True, proj_dim=128
    )
    ppg_model = PPG2ECGModel(ecg_ae=ecg_ae, cfg=ppg_cfg).to(DEVICE)
    ckpt_ppg = torch.load(PHASE1_PPG_PATH, map_location=DEVICE)
    ppg_model.load_state_dict(ckpt_ppg.get("model_state_dict", ckpt_ppg))
    ppg_model.eval()
    for p in ppg_model.parameters(): p.requires_grad = False

    # 3. Load Stage 2 Latent Rectified Flow
    flow_model = LatentRectifiedFlow(latent_channels=16, cond_channels=16, hidden_dim=128, num_blocks=6).to(DEVICE)
    flow_model.load_state_dict(torch.load(PHASE2_FLOW_PATH, map_location=DEVICE))
    flow_model.eval()
    for p in flow_model.parameters(): p.requires_grad = False
    
    return ecg_ae, ppg_model, flow_model

def euler_solve(flow_model, z_ppg, num_steps=10):
    """ Hàm giải ODE sinh predicted_ecg_latent từ z_ppg """
    B, C, L = z_ppg.shape
    xt = torch.randn((B, C, L), device=DEVICE) 
    dt = 1.0 / num_steps
    
    for step in range(num_steps):
        t_val = step * dt
        t_tensor = torch.full((B,), t_val, device=DEVICE)
        v_pred = flow_model(xt, t_tensor, z_ppg)
        xt = xt + v_pred * dt
        
    return xt

def align_signals(true_s: np.ndarray, pred_s: np.ndarray) -> np.ndarray:
    """ 
    Tìm độ lệch pha và dịch chuyển pred_s cho khớp với true_s bằng Cross-Correlation.
    """
    correlation = correlate(true_s, pred_s, mode='full')
    lag = np.argmax(correlation) - (len(pred_s) - 1)
    
    aligned_pred = np.empty_like(pred_s) 
    
    if lag > 0:
        aligned_pred[lag:] = pred_s[:-lag]
        edge_value = pred_s[0]
        aligned_pred[:lag] = edge_value
    elif lag < 0:
        lag = abs(lag)
        aligned_pred[:-lag] = pred_s[lag:]
        edge_value = pred_s[-1]
        aligned_pred[-lag:] = edge_value
    else:
        aligned_pred = pred_s.copy()
        
    return aligned_pred

# ==========================================
# CÁC CHỨC NĂNG CHÍNH (VISUALIZE, LOSS, SAVE, EVAL)
# ==========================================

def run_visualization():
    set_seed(SEED)
    ecg_ae, ppg_model, flow_model = load_models()

    try:
        dataset = LoadData(TEST_DATA_PATH)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print("[*] Displaying Reconstructed ECG vs Ground Truth for ALL records...")
    
    with torch.no_grad():
        for batch_data in dataloader:
            ecg = batch_data[0].float().unsqueeze(1).to(DEVICE) if batch_data[0].dim() == 2 else batch_data[0].float().to(DEVICE)
            ppg = batch_data[1].float().unsqueeze(1).to(DEVICE) if batch_data[1].dim() == 2 else batch_data[1].float().to(DEVICE)
            record_names = batch_data[2] if len(batch_data) > 2 else [f"Record_{i}" for i in range(ecg.shape[0])]

            feat_ppg = ppg_model.ppg_encoder(ppg)
            z_ppg = ppg_model.ppg_latent_head(feat_ppg)
            z_ecg_hat = euler_solve(flow_model, z_ppg, num_steps=ODE_STEPS)
            predicted_ecg = ecg_ae.decoder(z_ecg_hat)

            for i in range(ppg.shape[0]):
                ppg_plot = ppg[i].squeeze().cpu().numpy()
                ecg_gt_plot = ecg[i].squeeze().cpu().numpy()
                ecg_pred_plot = predicted_ecg[i].squeeze().cpu().numpy()
                
                t = np.arange(len(ecg_gt_plot))
                
                plt.figure(figsize=(12, 8))
                plt.suptitle(f"Record: {record_names[i]}", fontsize=14, fontweight='bold')

                plt.subplot(3, 1, 1)
                plt.plot(t, ppg_plot, color='green', label='Input PPG')
                plt.title("Input PPG Signal")
                plt.legend(loc='upper right')

                plt.subplot(3, 1, 2)
                plt.plot(t, ecg_gt_plot, color='blue', label='Ground Truth ECG')
                plt.title("Ground Truth ECG")
                plt.legend(loc='upper right')

                plt.subplot(3, 1, 3)
                plt.plot(t, ecg_gt_plot, color='black', label='Ground Truth', alpha=0.5)
                plt.plot(t, ecg_pred_plot, color='red', label='Predicted ECG (from Flow)', linestyle='--', alpha=0.8)
                plt.title("Comparison: Ground Truth vs Predicted ECG")
                plt.legend(loc='upper right')
                
                plt.tight_layout()
                plt.show() 

def run_loss():
    set_seed(SEED)
    ecg_ae, ppg_model, flow_model = load_models()
    
    try:
        test_dataset = LoadData(TEST_DATA_PATH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    except Exception as e:
        print(f"Error: {e}")
        return

    metrics = {
        'before': {'rmse': 0, 'pearson': 0, 'dtw': 0, 'cosine': 0},
        'after':  {'rmse': 0, 'pearson': 0, 'dtw': 0, 'cosine': 0}
    }
    total_samples = 0

    print("[*] Calculating Metrics...")
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Evaluating Metrics"):
            ecg = batch_data[0].float().unsqueeze(1).to(DEVICE) if batch_data[0].dim() == 2 else batch_data[0].float().to(DEVICE)
            ppg = batch_data[1].float().unsqueeze(1).to(DEVICE) if batch_data[1].dim() == 2 else batch_data[1].float().to(DEVICE)

            feat_ppg = ppg_model.ppg_encoder(ppg)
            z_ppg = ppg_model.ppg_latent_head(feat_ppg)
            z_ecg_hat = euler_solve(flow_model, z_ppg, num_steps=ODE_STEPS)
            predicted_ecg = ecg_ae.decoder(z_ecg_hat)

            ecg_true_np = np.atleast_2d(ecg.cpu().squeeze().numpy())
            ecg_pred_np = np.atleast_2d(predicted_ecg.cpu().squeeze().numpy())

            for b in range(ecg_true_np.shape[0]):
                true_s = ecg_true_np[b]
                pred_s = ecg_pred_np[b]

                rmse_b, pearson_b = calculate_metrics(true_s, pred_s)
                dtw_b = calculate_dtw_distance(true_s, pred_s)
                cosine_b = calculate_cosine_similarity(true_s, pred_s)
                
                metrics['before']['rmse'] += rmse_b
                metrics['before']['pearson'] += pearson_b
                metrics['before']['dtw'] += dtw_b
                metrics['before']['cosine'] += cosine_b

                aligned_pred_s = align_signals(true_s, pred_s)

                rmse_a, pearson_a = calculate_metrics(true_s, aligned_pred_s)
                dtw_a = calculate_dtw_distance(true_s, aligned_pred_s)
                cosine_a = calculate_cosine_similarity(true_s, aligned_pred_s)

                metrics['after']['rmse'] += rmse_a
                metrics['after']['pearson'] += pearson_a
                metrics['after']['dtw'] += dtw_a
                metrics['after']['cosine'] += cosine_a
                
                total_samples += 1

    print(f"\n{'='*40}")
    print(f"FINAL RESULTS ({total_samples} samples)")
    print(f"{'='*40}")
    print(f"{'Metric':<12} | {'Before Align':<12} | {'After Align':<12}")
    print(f"{'-'*40}")
    print(f"RMSE         | {metrics['before']['rmse']/total_samples:<12.4f} | {metrics['after']['rmse']/total_samples:<12.4f}")
    print(f"Pearson      | {metrics['before']['pearson']/total_samples:<12.4f} | {metrics['after']['pearson']/total_samples:<12.4f}")
    print(f"DTW          | {metrics['before']['dtw']/total_samples:<12.4f} | {metrics['after']['dtw']/total_samples:<12.4f}")
    print(f"Cosine       | {metrics['before']['cosine']/total_samples:<12.4f} | {metrics['after']['cosine']/total_samples:<12.4f}")
    print(f"{'='*40}")

def save_ecg_reconstruction(output_path="AF_Detection/ecg_reconstructions.npz"):
    set_seed(SEED)
    ecg_ae, ppg_model, flow_model = load_models()
    
    all_predicted_ecgs = []
    all_original_ppgs = []
    all_labels = []
    all_record_names = []

    try:
        test_dataset = LoadData(TEST_DATA_PATH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    except Exception as e:
        print(f"Error: {e}")
        return
   
    print(f"[*] Running inference and saving reconstructions to {output_path}...")
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Saving Data"):
            ecg = batch_data[0]
            ppg = batch_data[1]
            record_names = batch_data[2] if len(batch_data) > 2 else []
            label = batch_data[3] if len(batch_data) > 3 else np.zeros(len(ecg))

            ppg_input = ppg.to(DEVICE).float().unsqueeze(1) if ppg.dim() == 2 else ppg.float().to(DEVICE)
            
            feat_ppg = ppg_model.ppg_encoder(ppg_input)
            z_ppg = ppg_model.ppg_latent_head(feat_ppg)
            z_ecg_hat = euler_solve(flow_model, z_ppg, num_steps=ODE_STEPS)
            
            predicted_ecg = ecg_ae.decoder(z_ecg_hat)
            
            all_predicted_ecgs.append(predicted_ecg.squeeze(1).cpu().numpy())
            all_original_ppgs.append(ppg.cpu().numpy())
            all_labels.append(label.cpu().numpy() if isinstance(label, torch.Tensor) else label)
            all_record_names.append(record_names)
            
    save_dict = {
        "ecgs": np.concatenate(all_predicted_ecgs, axis=0),
        "ppgs": np.concatenate(all_original_ppgs, axis=0),
        "labels": np.concatenate(all_labels, axis=0),
        "records": np.concatenate(all_record_names, axis=0) if all_record_names else np.array([])
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, **save_dict)
    print(f"[+] Saved reconstructions successfully to {output_path}")

def run_peak_count_evaluation():
    set_seed(SEED)
    ecg_ae, ppg_model, flow_model = load_models()
    
    try:
        test_dataset = LoadData(TEST_DATA_PATH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    except Exception as e:
        print(f"Error: {e}")
        return

    total_true_peaks = 0
    total_pred_peaks = 0
    total_ratio = 0.0
    total_samples = 0

    print("[*] Đang đếm số lượng đỉnh R...")
    print("[!] CẢNH BÁO: Đang ép Predicted ECG dùng chung ngưỡng với Ground Truth!")
    
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Counting R-Peaks"):
            ecg = batch_data[0].float().unsqueeze(1).to(DEVICE) if batch_data[0].dim() == 2 else batch_data[0].float().to(DEVICE)
            ppg = batch_data[1].float().unsqueeze(1).to(DEVICE) if batch_data[1].dim() == 2 else batch_data[1].float().to(DEVICE)

            feat_ppg = ppg_model.ppg_encoder(ppg)
            z_ppg = ppg_model.ppg_latent_head(feat_ppg)
            z_ecg_hat = euler_solve(flow_model, z_ppg, num_steps=ODE_STEPS)
            predicted_ecg = ecg_ae.decoder(z_ecg_hat)

            ecg_true_np = np.atleast_2d(ecg.cpu().squeeze().numpy())
            ecg_pred_np = np.atleast_2d(predicted_ecg.cpu().squeeze().numpy())

            for b in range(ecg_true_np.shape[0]):
                true_s = ecg_true_np[b]
                pred_s = ecg_pred_np[b]

                num_true, num_pred, ratio = calculate_peak_count_ratio(true_s, pred_s, sampling_rate=125)
                
                total_true_peaks += num_true
                total_pred_peaks += num_pred
                total_ratio += ratio
                total_samples += 1

    # Tính toán kết quả trung bình
    avg_ratio = total_ratio / total_samples if total_samples > 0 else 0
    
    mae_peaks = abs(total_true_peaks - total_pred_peaks) / total_samples if total_samples > 0 else 0

    print(f"\n{'='*50}")
    print(f"BÁO CÁO SỐ LƯỢNG ĐỈNH R ({total_samples} samples)")
    print(f"{'='*50}")
    print(f"Tổng số đỉnh R thực tế (Ground Truth) : {total_true_peaks}")
    print(f"Tổng số đỉnh R mô hình sinh ra (Pred) : {total_pred_peaks}")
    print(f"Tỷ lệ số đỉnh trung bình (Pred/True)  : {avg_ratio * 100:.2f} %")
    print(f"Sai lệch trung bình trên mỗi tín hiệu : {mae_peaks:.2f} đỉnh/tín hiệu")
    print(f"{'='*50}")

if __name__ == "__main__":
    # CHỌN 1 TRONG 3 HÀM ĐỂ CHẠY (Bỏ comment để sử dụng):
    
    # run_visualization()
    # run_loss()
    save_ecg_reconstruction("/home/linhhima/PPG_ECG/AF_Detection/total_mimic_af_recon.npz")
    # run_peak_count_evaluation()