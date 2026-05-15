import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from load_data import LoadData
from CLIP import ECG_PPG_Fusion_Model  
import random
import os
from scipy.signal import butter, filtfilt, find_peaks
from metric import calculate_cosine_similarity, calculate_dtw_distance, calculate_metrics
from scipy.signal import correlate
from tqdm import tqdm

# --- CONFIGURATION ---
SEED = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
INPUT_LENGTH = 2400
OUTPUT_EMBED_DIM = 128
SEQ_LENGTH = 2400
TEST_DATA_PATH = '/home/linhhima/PPG_ECG/datasets/z_score_norm/total_mimic_af.npz'
CLIP_MODEL_PATH = '/home/linhhima/PPG_ECG/best_multitask_model.pth'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def pan_tompkins_qrs(ecg_signal: np.ndarray, fs: int = 125, external_threshold: float = None, return_threshold: bool = False):
    """
    Thuật toán Pan-Tompkins đã chỉnh sửa để nhận/trả ngưỡng.
    """
    ecg_signal = np.array(ecg_signal).flatten()
    
    # 1. Bandpass Filter
    nyq = 0.5 * fs
    low = 5.0 / nyq
    high = 15.0 / nyq
    b, a = butter(1, [low, high], btype='band')
    filtered_ecg = filtfilt(b, a, ecg_signal)
    
    # 2. Derivative
    diff_ecg = np.diff(filtered_ecg)
    diff_ecg = np.insert(diff_ecg, 0, diff_ecg[0])
    
    # 3. Squaring
    squared_ecg = diff_ecg ** 2
    
    # 4. Moving Window Integration
    window_width = int(0.15 * fs)
    integrated_ecg = np.convolve(squared_ecg, np.ones(window_width) / window_width, mode='same')
    
    # 5. THRESHOLDING
    # Nếu có truyền ngưỡng từ ngoài vào thì dùng nó, nếu không thì tự tính trung bình
    if external_threshold is not None:
        threshold = external_threshold
    else:
        threshold = np.mean(integrated_ecg)
        
    min_distance = int(0.3 * fs)
    peaks_integrated, _ = find_peaks(integrated_ecg, height=threshold, distance=min_distance)
    
    # 6. Back-search
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
    
    # 4. Tính tỷ lệ dự đoán so với thực tế (Prediction Ratio)
    if num_true > 0:
        ratio = num_pred / num_true
    else:
        ratio = 1.0 if num_pred == 0 else 0.0
        
    return num_true, num_pred, ratio

def load_model():
    print(f"Loading model on {DEVICE}...")
    
    model = ECG_PPG_Fusion_Model(embed_dim=OUTPUT_EMBED_DIM, freq_dim=256, target_length=SEQ_LENGTH).to(DEVICE)

    if torch.cuda.is_available():
        weights = torch.load(CLIP_MODEL_PATH)
    else:
        weights = torch.load(CLIP_MODEL_PATH, map_location='cpu')

    model.load_state_dict(weights)
    model.eval()
    
    return model

def save_ecg_reconstruction(output_path="AF_Detection/ecg_reconstructions.npz"):
    set_seed(SEED)
    model = load_model()
    
    all_predicted_ecgs = []
    all_original_ppgs = []
    all_labels = []
    all_record_names = []

    try:
        test_dataset = LoadData(TEST_DATA_PATH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    except Exception as e:
        print(f"Error: {e}")
        return
   
    print("Running inference and saving reconstructions...")
    with torch.no_grad():
        for i, (ecg, ppg, record_names, label) in enumerate(test_loader):
            ppg_input = ppg.to(DEVICE).float().unsqueeze(1)
            ecg_input = ecg.to(DEVICE).float().unsqueeze(1)

            _,_,predicted_ecg = model(ecg_input,ppg_input)
            
            all_predicted_ecgs.append(predicted_ecg.squeeze(1).cpu().numpy())
            all_original_ppgs.append(ppg.cpu().numpy())
            all_labels.append(label.cpu().numpy())
            all_record_names.append(record_names)
            
    save_dict = {
        "ecgs": np.concatenate(all_predicted_ecgs, axis=0),
        "ppgs": np.concatenate(all_original_ppgs, axis=0),
        "labels": np.concatenate(all_labels, axis=0),
        "records": np.array(all_record_names, dtype=object) 
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, **save_dict)
    print(f"Saved reconstructions to {output_path}")

def visualize_results(ppg, ecg_true, ecg_pred, record_name):
    print(f"Visualizing Record: {record_name}")
    t_ppg = np.arange(len(ppg))
    t_ecg = np.arange(len(ecg_true))

    plt.figure(figsize=(12, 10))
    plt.suptitle(f"Record: {record_name}", fontsize=14, fontweight='bold')

    plt.subplot(4, 1, 1)
    plt.plot(t_ppg, ppg, color='green', label='Input PPG')
    plt.title("Input PPG Signal")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    plt.subplot(4, 1, 2)
    plt.plot(t_ecg, ecg_true, color='blue', label='Ground Truth ECG')
    plt.title("Ground Truth ECG")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    plt.subplot(4, 1, 3)
    plt.plot(t_ecg, ecg_pred, color='red', label='Predicted ECG')
    plt.title("Predicted ECG (Reconstructed)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    plt.subplot(4, 1, 4)
    plt.plot(t_ecg, ecg_true, color='black', label='Ground Truth', alpha=0.7)
    plt.plot(t_ecg, ecg_pred, color='red', label='Predicted', linestyle='--', alpha=0.8)
    plt.title("Comparison: Ground Truth vs Predicted ECG")
    plt.xlabel("Time Samples")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def run_visualization():
    set_seed(SEED)
    model = load_model()
    seen_records = set()

    try:
        test_dataset = LoadData(TEST_DATA_PATH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    except Exception as e:
        print(f"Error: {e}")
        return
   
    print("Running inference for visualization...")
    with torch.no_grad():
        for i, (ecg, ppg, record_names) in enumerate(test_loader):
            ppg_input = ppg.to(DEVICE).float().unsqueeze(1)
            ecg_input = ecg.to(DEVICE).float().unsqueeze(1)

            _, _, predicted_ecg = model(ecg_input, ppg_input)

            ppg_np = np.atleast_2d(ppg_input.cpu().squeeze().numpy())
            ecg_true_np = np.atleast_2d(ecg.cpu().squeeze().numpy())
            ecg_pred_np = np.atleast_2d(predicted_ecg.cpu().squeeze().numpy())

            for idx in range(ppg_np.shape[0]):
                current_rec_name = record_names[idx]
                
                if current_rec_name not in seen_records:
                    visualize_results(
                        ppg_np[idx], 
                        ecg_true_np[idx], 
                        ecg_pred_np[idx], 
                        current_rec_name
                    )
                    seen_records.add(current_rec_name)

def align_signals(true_s: np.ndarray, pred_s: np.ndarray) -> np.ndarray:
    """
    Dùng Cross-Correlation để tìm độ lệch pha và dịch chuyển pred_s cho khớp với true_s.
    """
    correlation = correlate(true_s, pred_s, mode='full')
    lag = np.argmax(correlation) - (len(pred_s) - 1)
    
    aligned_pred = np.zeros_like(pred_s)
    
    if lag > 0:
        aligned_pred[lag:] = pred_s[:-lag]
    elif lag < 0:
        aligned_pred[:lag] = pred_s[-lag:]
    else:
        aligned_pred = pred_s.copy()
        
    return aligned_pred

def run_loss():
    model = load_model()
    
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

    print("Calculating Metrics...")
    with torch.no_grad():
        for i, (ecg, ppg, record_names) in enumerate(test_loader):
            ppg_input = ppg.to(DEVICE).float().unsqueeze(1)
            ecg_input = ecg.to(DEVICE).float().unsqueeze(1)

            _,_, predicted_ecg = model(ecg_input, ppg_input)

            ecg_true_np = np.atleast_2d(ecg.cpu().squeeze().numpy())
            ecg_pred_np = np.atleast_2d(predicted_ecg.cpu().squeeze().numpy())

            for b in range(ecg_true_np.shape[0]):
                true_s = ecg_true_np[b]
                pred_s = ecg_pred_np[b]

                # Trước Align
                rmse_b, pearson_b = calculate_metrics(true_s, pred_s)
                dtw_b = calculate_dtw_distance(true_s, pred_s)
                cosine_b = calculate_cosine_similarity(true_s, pred_s)
                
                metrics['before']['rmse'] += rmse_b
                metrics['before']['pearson'] += pearson_b
                metrics['before']['dtw'] += dtw_b
                metrics['before']['cosine'] += cosine_b

                # Align
                aligned_pred_s = align_signals(true_s, pred_s)

                # Sau Align
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

def run_peak_count_evaluation():
    set_seed(SEED)
    model = load_model()
    
    try:
        test_dataset = LoadData(TEST_DATA_PATH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) 
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        return

    total_true_peaks = 0
    total_pred_peaks = 0
    total_ratio = 0.0
    total_samples = 0

    print("[*] Đang đếm số lượng đỉnh R bằng thuật toán Pan-Tompkins...")
    print("[!] ĐANG DÙNG NGƯỠNG CỦA GROUND TRUTH ÉP CHO PREDICTED ECG")
    
    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(test_loader, desc="Counting R-Peaks")):
            ecg = batch_data[0]
            ppg = batch_data[1]
            
            ppg_input = ppg.to(DEVICE).float().unsqueeze(1)
            ecg_input = ecg.to(DEVICE).float().unsqueeze(1)

            _, _, predicted_ecg = model(ecg_input, ppg_input)

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
    # run_visualization()
    # run_loss()
    # run_peak_count_evaluation()
    save_ecg_reconstruction(output_path = '/home/linhhima/PPG_ECG/AF_Detection/total_mimic_af_recon.npz')