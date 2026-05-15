import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy.signal import butter, filtfilt, find_peaks
from tqdm import tqdm  # Thêm thư viện tqdm để tạo thanh tiến trình
import random
import os

# Import từ thư viện của bạn
from load_data import LoadData
from CLIP import ECG_PPG_Fusion_Model 

# ==========================================
# 1. CẤU HÌNH (CONFIGURATIONS)
# ==========================================
SEED = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
INPUT_LENGTH = 2400
OUTPUT_EMBED_DIM = 128
SEQ_LENGTH = 2400
TEST_DATA_PATH = '/home/linhhima/PPG_ECG/datasets/z_score_norm/total_mimic_af.npz'
CLIP_MODEL_PATH = '/home/linhhima/PPG_ECG/best_multitask_model.pth'

SAMPLING_RATE = 125 # Hz
LEFT_WINDOW_MS = 250  # Lùi về trước 250ms để lấy sóng P
RIGHT_WINDOW_MS = 400 # Tiến về sau 400ms để lấy sóng T

# ==========================================
# 2. CÁC HÀM TIỆN ÍCH
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_model():
    print(f"[*] Loading model on {DEVICE}...")
    model = ECG_PPG_Fusion_Model(embed_dim=OUTPUT_EMBED_DIM, freq_dim=256, target_length=SEQ_LENGTH).to(DEVICE)
    weights = torch.load(CLIP_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(weights)
    model.eval()
    return model

def pan_tompkins_qrs(ecg_signal: np.ndarray, fs: int = 125):
    """Thuật toán Pan-Tompkins tự động tìm đỉnh R độc lập cho mỗi tín hiệu"""
    ecg_signal = np.array(ecg_signal).flatten()
    nyq = 0.5 * fs
    b, a = butter(1, [5.0 / nyq, 15.0 / nyq], btype='band')
    filtered_ecg = filtfilt(b, a, ecg_signal)
    
    diff_ecg = np.diff(filtered_ecg)
    diff_ecg = np.insert(diff_ecg, 0, diff_ecg[0])
    squared_ecg = diff_ecg ** 2
    
    window_width = int(0.15 * fs)
    integrated_ecg = np.convolve(squared_ecg, np.ones(window_width) / window_width, mode='same')
    
    threshold = np.mean(integrated_ecg)
    min_distance = int(0.3 * fs)
    peaks_integrated, _ = find_peaks(integrated_ecg, height=threshold, distance=min_distance)
    
    r_peaks = []
    search_window = int(0.05 * fs)
    for p in peaks_integrated:
        start = max(0, p - search_window)
        end = min(len(ecg_signal), p + search_window)
        if start < end:
            local_max = np.argmax(ecg_signal[start:end])
            r_peaks.append(start + local_max)
            
    return np.array(r_peaks)

def extract_heartbeats(ecg_signal: np.ndarray, r_peaks: np.ndarray, fs: int, left_ms: int, right_ms: int):
    """Cắt các cụm P-QRS-T xung quanh đỉnh R"""
    ecg_signal = np.array(ecg_signal).flatten()
    left_samples = int((left_ms / 1000.0) * fs)
    right_samples = int((right_ms / 1000.0) * fs)
    
    heartbeats = []
    for r in r_peaks:
        if r - left_samples >= 0 and r + right_samples < len(ecg_signal):
            beat = ecg_signal[r - left_samples : r + right_samples]
            heartbeats.append(beat)
            
    return np.array(heartbeats)

def calculate_complex_metrics(true_complex, pred_complex):
    """Tính RMSE và Pearson cho 1 cụm P-QRS-T"""
    # 1. RMSE
    rmse = np.sqrt(np.mean((true_complex - pred_complex) ** 2))
    
    # 2. Pearson Correlation
    std_true = np.std(true_complex)
    std_pred = np.std(pred_complex)
    
    if std_true < 1e-6 or std_pred < 1e-6:
        pearson = 0.0 
    else:
        pearson = np.corrcoef(true_complex, pred_complex)[0, 1]
        
    return rmse, pearson

# ==========================================
# 3. HÀM CHÍNH: TRÍCH XUẤT VÀ TÍNH TRUNG BÌNH
# ==========================================
def run_single_complex_evaluation():
    set_seed(SEED)
    model = load_model()
    seen_records = set() # Set dùng để theo dõi các record đã vẽ
    
    # CỜ KIỂM SOÁT ĐỒ THỊ (Để False nếu chỉ muốn lấy điểm trung bình nhanh)
    SHOW_PLOTS = True
    
    # Các biến cộng dồn để tính trung bình
    total_rmse = 0.0
    total_pearson = 0.0
    valid_record_count = 0
    
    try:
        test_dataset = LoadData(TEST_DATA_PATH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) 
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print("[*] Bắt đầu trích xuất 1 cụm P-QRS-T cho mỗi Record...")
    if not SHOW_PLOTS:
        print("[!] Chế độ ẩn đồ thị được bật (SHOW_PLOTS = False). Đang tính toán điểm trung bình...")
    
    # Tạo trục thời gian cho đồ thị (từ -250ms đến +400ms)
    total_samples = int((LEFT_WINDOW_MS + RIGHT_WINDOW_MS) / 1000.0 * SAMPLING_RATE)
    time_axis_ms = np.linspace(-LEFT_WINDOW_MS, RIGHT_WINDOW_MS, total_samples)

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Processing Batches")):
            ecg = batch_data[0].float().unsqueeze(1).to(DEVICE) if batch_data[0].dim() == 2 else batch_data[0].float().to(DEVICE)
            ppg = batch_data[1].float().unsqueeze(1).to(DEVICE) if batch_data[1].dim() == 2 else batch_data[1].float().to(DEVICE)
            
            record_names = batch_data[2] if len(batch_data) > 2 else [f"Batch_{batch_idx}_Idx_{i}" for i in range(ecg.shape[0])]

            _, _, predicted_ecg = model(ecg, ppg)

            ecg_true_np = ecg.cpu().squeeze().numpy()
            ecg_pred_np = predicted_ecg.cpu().squeeze().numpy()

            ecg_true_np = np.atleast_2d(ecg_true_np)
            ecg_pred_np = np.atleast_2d(ecg_pred_np)

            for b in range(ecg_true_np.shape[0]):
                rec_name = record_names[b]
                
                # Bỏ qua nếu record này đã được xử lý rồi
                if rec_name in seen_records:
                    continue

                true_s = ecg_true_np[b]
                pred_s = ecg_pred_np[b]

                # Bước 1: Tìm đỉnh R độc lập
                true_peaks = pan_tompkins_qrs(true_s, fs=SAMPLING_RATE)
                pred_peaks = pan_tompkins_qrs(pred_s, fs=SAMPLING_RATE)

                if len(true_peaks) == 0 or len(pred_peaks) == 0:
                    continue

                # Bước 2: Cắt các chu kỳ tim
                true_beats = extract_heartbeats(true_s, true_peaks, SAMPLING_RATE, LEFT_WINDOW_MS, RIGHT_WINDOW_MS)
                pred_beats = extract_heartbeats(pred_s, pred_peaks, SAMPLING_RATE, LEFT_WINDOW_MS, RIGHT_WINDOW_MS)

                if len(true_beats) == 0 or len(pred_beats) == 0:
                    continue

                # Bước 3: Chỉ lấy đúng 1 cụm duy nhất (cụm đầu tiên hợp lệ)
                true_complex = true_beats[0]
                pred_complex = pred_beats[0]

                # Đánh dấu record này đã được xử lý
                seen_records.add(rec_name)

                # Bước 4: Tính toán RMSE và Pearson trên 1 cụm này
                rmse_val, pearson_val = calculate_complex_metrics(true_complex, pred_complex)
                
                # Cộng dồn điểm số
                total_rmse += rmse_val
                total_pearson += pearson_val
                valid_record_count += 1

                # Bước 5: Vẽ đồ thị so sánh trực tiếp (Chỉ vẽ khi SHOW_PLOTS = True)
                if SHOW_PLOTS:
                    plt.figure(figsize=(10, 6))
                    plt.plot(time_axis_ms, true_complex, color='black', linewidth=2.5, label='Ground Truth Complex')
                    plt.plot(time_axis_ms, pred_complex, color='red', linestyle='--', linewidth=2, label='Predicted Complex')
                    
                    plt.axvline(x=0, color='gray', linestyle=':', alpha=0.7, label='R-peak (0 ms)')

                    plt.title(f"Record: {rec_name}\nPPG2ECG Model: Single P-QRS-T Complex", fontsize=14, fontweight='bold')
                    plt.suptitle(f"RMSE: {rmse_val:.4f}  |  Pearson: {pearson_val:.4f}", color='blue', fontsize=12)
                    plt.xlabel("Time relative to R-peak (ms)", fontsize=11)
                    plt.ylabel("Amplitude", fontsize=11)
                    plt.legend(loc="upper right")
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.show()

    # IN BÁO CÁO KẾT QUẢ TRUNG BÌNH TỔNG THỂ
    if valid_record_count > 0:
        avg_rmse = total_rmse / valid_record_count
        avg_pearson = total_pearson / valid_record_count
        
        print(f"\n{'='*55}")
        print(f"BÁO CÁO KẾT QUẢ TRUNG BÌNH (CLIP MODEL) - P-QRS-T TEMPLATE")
        print(f"{'='*55}")
        print(f"Tổng số Record hợp lệ được đánh giá : {valid_record_count}")
        print(f"Average RMSE                       : {avg_rmse:.4f}")
        print(f"Average Pearson Correlation        : {avg_pearson:.4f}")
        print(f"{'='*55}")
    else:
        print("\n[!] Không tìm thấy đoạn P-QRS-T hợp lệ nào trong toàn bộ tập dữ liệu.")

if __name__ == "__main__":
    run_single_complex_evaluation()