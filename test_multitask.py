import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from load_data import LoadData
from CLIP import ECGDecoder_UNet, ECGEssembleCLIP, PPGtoECGConverter
import random
import os
from scipy import signal

from metric import calculate_cosine_similarity, calculate_dtw_distance, calculate_metrics, calculate_prd, calculate_ssim_1d

# --- CONFIGURATION ---
SEED = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16 # Small batch for testing
INPUT_LENGTH = 2400
OUTPUT_EMBED_DIM = 128
TEST_DATA_PATH = 'datasets/total_record_z_test.npz' # Change to normal_train.npz if val doesn't exist yet
CLIP_MODEL_PATH = "multitask_clip_best_model_total_record_z.pth"
DECODER_MODEL_PATH = "multitask_decoder_best_model_total_record_z.pth"
NUM_SAMPLES_TO_PLOT = 10

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_models():
    """Initializes models and loads the trained weights."""
    print(f"Loading models on {DEVICE}...")
    
    # 1. Initialize Architecture
    model_clip = ECGEssembleCLIP(embed_dim=OUTPUT_EMBED_DIM).to(DEVICE)
    model_converter = ECGDecoder_UNet(
        bottleneck_channels=2048
    ).to(DEVICE)

    # 2. Load Weights
    # Load CLIP (Encoder)
    if torch.cuda.is_available():
        clip_weights = torch.load(CLIP_MODEL_PATH)
        decoder_weights = torch.load(DECODER_MODEL_PATH)
    else:
        clip_weights = torch.load(CLIP_MODEL_PATH, map_location=torch.device('cpu'))
        decoder_weights = torch.load(DECODER_MODEL_PATH, map_location=torch.device('cpu'))

    model_clip.load_state_dict(clip_weights)
    
    # CRITICAL: In train.py, you saved only model_converter.ecg_decoder
    model_converter.load_state_dict(decoder_weights)
    
    model_clip.eval()
    model_converter.eval()
    
    return model_clip, model_converter

def check_phase_shift(ecg_true, ecg_pred, fs=100, record_name="Unknown"):
    """
    Tính toán độ lệch pha (Phase Shift) giữa tín hiệu gốc và dự đoán.
    
    Args:
        ecg_true (np.array): Tín hiệu ECG gốc (1D array).
        ecg_pred (np.array): Tín hiệu ECG dự đoán (1D array).
        fs (int): Tần số lấy mẫu (sampling rate), dùng để tính thời gian lệch.
    """
    # 1. Chuẩn hóa tín hiệu (Zero-mean) để loại bỏ ảnh hưởng của DC offset
    true_norm = ecg_true - np.mean(ecg_true)
    pred_norm = ecg_pred - np.mean(ecg_pred)
    
    # 2. Tính Cross-Correlation
    # mode='full' trả về kết quả có độ dài len(true) + len(pred) - 1
    correlation = signal.correlate(true_norm, pred_norm, mode='full')
    
    # 3. Tìm vị trí (index) có độ tương đồng cao nhất (đỉnh của correlation)
    lags = signal.correlation_lags(len(true_norm), len(pred_norm), mode='full')
    max_corr_index = np.argmax(correlation)
    lag_samples = lags[max_corr_index] # Số mẫu bị lệch
    
    # Tính thời gian lệch (giây)
    time_shift = lag_samples / fs
    
    print(f"--- Phase Shift Analysis for {record_name} ---")
    print(f"Lag (Samples): {lag_samples}")
    print(f"Time Shift: {time_shift:.4f} seconds")
    
    if lag_samples == 0:
        print(">> KẾT LUẬN: Đồng bộ hoàn hảo (No phase shift).")
    elif lag_samples > 0:
        print(f">> KẾT LUẬN: ecg_pred đi SỚM hơn ecg_true {lag_samples} mẫu (Shift Left).")
    else:
        print(f">> KẾT LUẬN: ecg_pred đi TRỄ hơn ecg_true {abs(lag_samples)} mẫu (Shift Right).")

    # 4. Vẽ biểu đồ minh họa
    plt.figure(figsize=(10, 6))
    
    # Plot 1: Tín hiệu gốc vs Dự đoán
    plt.subplot(2, 1, 1)
    plt.plot(ecg_true, label='ECG True', color='black')
    plt.plot(ecg_pred, label='ECG Pred', color='red', linestyle='--')
    plt.title(f'Original Signals (Lag: {lag_samples} samples)')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Tín hiệu sau khi đã sửa lệch pha (Shifted correction)
    # Dịch chuyển ecg_pred để khớp với ecg_true
    plt.subplot(2, 1, 2)
    if lag_samples > 0:
        # Pred đi sớm -> cần đẩy lùi ra sau (chèn 0 vào đầu)
        ecg_pred_corrected = np.pad(ecg_pred, (lag_samples, 0), 'constant')[:len(ecg_true)]
    elif lag_samples < 0:
         # Pred đi trễ -> cần cắt bớt đầu
        ecg_pred_corrected = np.pad(ecg_pred, (0, abs(lag_samples)), 'constant')[abs(lag_samples):]
    else:
        ecg_pred_corrected = ecg_pred

    plt.plot(ecg_true, label='ECG True', color='black')
    plt.plot(ecg_pred_corrected, label='ECG Pred (Corrected)', color='green', linestyle='--')
    plt.title('Signals after shifting correction')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def check_phase_shift_and_error(ecg_true, ecg_pred, fs=125, record_name="Unknown"):
    """
    Tính toán độ lệch pha, sửa tín hiệu và hiển thị Error Signal sau khi sửa.
    """
    # Đảm bảo input là numpy array 1D
    ecg_true = np.array(ecg_true).flatten()
    ecg_pred = np.array(ecg_pred).flatten()

    true_norm = ecg_true 
    pred_norm = ecg_pred 
    
    # 2. Tính Cross-Correlation
    correlation = signal.correlate(true_norm, pred_norm, mode='full')
    lags = signal.correlation_lags(len(true_norm), len(pred_norm), mode='full')
    
    # 3. Tìm độ lệch (Lag) tốt nhất
    max_corr_index = np.argmax(correlation)
    lag_samples = lags[max_corr_index]
    time_shift = lag_samples / fs
    
    # 4. Tạo tín hiệu đã sửa pha (Corrected Prediction)
    # Logic: Giữ nguyên True, dịch chuyển Pred để khớp True
    
    if lag_samples > 0:
        # Pred đang đi SỚM (bên trái) -> Cần dịch sang PHẢI (chèn 0 vào đầu)
        ecg_pred_corrected = np.pad(ecg_pred, (lag_samples, 0), 'constant')[:len(ecg_true)]
        shift_msg = f"ECG prediction leads (early). Needs a right shift of {lag_samples} samples."
        
    elif lag_samples < 0:
        # Pred đang đi TRỄ (bên phải) -> Cần dịch sang TRÁI (cắt bớt đầu)
        
        # --- SỬA LỖI Ở ĐÂY: Tính abs_lag TRƯỚC khi dùng trong f-string ---
        abs_lag = abs(lag_samples) 
        # -----------------------------------------------------------------

        ecg_pred_corrected = np.pad(ecg_pred, (0, abs_lag), 'constant')[abs_lag:]
        shift_msg = f"ECG prediction lags (late). Needs a left shift of {abs_lag} samples."
        
    else:
        # Không lệch
        ecg_pred_corrected = ecg_pred
        shift_msg = "Perfectly synchronized (No phase shift)."

    # 5. Tính toán Error Signal
    error_before = ecg_true - ecg_pred
    error_after = ecg_true - ecg_pred_corrected 

    # --- IN KẾT QUẢ ---
    print(f"\n--- Phase Shift Analysis: {record_name} ---")
    print(f"Lag (Samples): {lag_samples}")
    print(f"Time Shift   : {time_shift:.4f}s")
    print(f"Correction   : {shift_msg}")
    print(f"MAE (Before) : {np.mean(np.abs(error_before)):.4f}")
    print(f"MAE (After)  : {np.mean(np.abs(error_after)):.4f}")

    # --- VẼ BIỂU ĐỒ ---
    plt.figure(figsize=(12, 10))
    t = np.arange(len(ecg_true))

    # Plot 1: Gốc (Chưa sửa)
    plt.subplot(3, 1, 1)
    plt.plot(t, ecg_true, 'k', label='Ground Truth', linewidth=1.5, alpha=0.7)
    plt.plot(t, ecg_pred, 'r--', label='Original Pred', linewidth=1.5)
    plt.title(f"1. Original Signals (Lag: {lag_samples})")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    # Plot 2: Đã sửa (Đồng bộ)
    plt.subplot(3, 1, 2)
    plt.plot(t, ecg_true, 'k', label='Ground Truth', linewidth=1.5, alpha=0.7)
    plt.plot(t, ecg_pred_corrected, 'g--', label='Corrected Pred', linewidth=1.5)
    plt.title(f"2. Phase Corrected Signals ({shift_msg})")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    # Plot 3: Error Signal
    plt.subplot(3, 1, 3)
    plt.plot(t, error_before, color='gray', alpha=0.3, label='Error Before Shift')
    plt.plot(t, error_after, color='purple', label='Error After Shift')
    plt.fill_between(t, error_after, color='purple', alpha=0.2)
    plt.title("3. Error Signal Comparison")
    plt.ylabel("Difference")
    plt.xlabel("Samples")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def visualize_results(ppg, ecg_true, ecg_pred, sample_idx, record_name):
    """
    Draws 4 subplots: PPG, True ECG, Predicted ECG, Error Signal.
    Includes record_name in the title.
    """
    
    error_signal = ecg_true - ecg_pred
    print(f"Visualizing results for Record: {record_name}")
    # print("ecg_true: ", ecg_true)
    # print("ecg_pred: ", ecg_pred)
    # print("error_signal: ", error_signal)
    # check_phase_shift_and_error(ecg_true, ecg_pred, fs=125, record_name=record_name)
    
    #################################################################################
    # t = np.arange(len(ppg))

    # plt.figure(figsize=(12, 10))
    
    # # Tiêu đề chính
    # plt.suptitle(f"Record: {record_name}", fontsize=16, fontweight='bold')

    # # 1. PPG Input (Giữ riêng ở trên cùng)
    # plt.subplot(3, 1, 1)
    # plt.plot(t, ppg, color='green', label='Input PPG', linewidth=1.5)
    # plt.title("Input PPG Signal")
    # plt.ylabel("Amplitude")
    # plt.legend(loc='upper right')
    # plt.grid(True, alpha=0.3)

    # # 2. Comparison: Ground Truth vs Predicted (CÙNG TRÊN 1 TRỤC)
    # plt.subplot(3, 1, 2)
    # plt.plot(t, ecg_true, color='black', label='Ground Truth ECG', linewidth=1.5, alpha=0.8)
    # plt.plot(t, ecg_pred, color='red', label='Predicted ECG', linestyle='--', linewidth=1.5)
    # plt.title("Comparison: Ground Truth vs Predicted ECG")
    # plt.ylabel("Amplitude")
    # plt.legend(loc='upper right')
    # plt.grid(True, alpha=0.3)

    # # 3. Error Signal (Giữ riêng ở dưới cùng để xem độ lệch)
    # plt.subplot(3, 1, 3)
    # plt.plot(t, error_signal, color='purple', label='Error (True - Pred)')
    # plt.fill_between(t, error_signal, color='purple', alpha=0.2) # Tô màu vùng lỗi
    # plt.title("Error Signal")
    # plt.xlabel("Time Samples")
    # plt.ylabel("Difference")
    # plt.legend(loc='upper right')
    # plt.grid(True, alpha=0.3)

    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()

    #################################################################################

    # # Calculate Error Signal (Difference)
    # # Create Time Axis (optional, assuming indices)
    # t = np.arange(len(ppg))

    # plt.figure(figsize=(12, 10))
    
    # # --- CẬP NHẬT: Thêm tên Record vào tiêu đề chính ---
    # plt.suptitle(f"Record: {record_name}", fontsize=16, fontweight='bold')

    # # 1. PPG Input
    # plt.subplot(4, 1, 1)
    # plt.plot(t, ppg, color='green', label='Input PPG')
    # plt.title("Input PPG Signal")
    # plt.ylabel("Amplitude")
    # plt.legend(loc='upper right')
    # plt.grid(True, alpha=0.3)

    # # 2. Ground Truth ECG
    # plt.subplot(4, 1, 2)
    # plt.plot(t, ecg_true, color='blue', label='Ground Truth ECG')
    # plt.title("Ground Truth ECG")
    # plt.ylabel("Amplitude")
    # plt.legend(loc='upper right')
    # plt.grid(True, alpha=0.3)

    # # 3. Predicted ECG
    # plt.subplot(4, 1, 3)
    # plt.plot(t, ecg_pred, color='red', label='Predicted ECG')
    # plt.title("Predicted ECG (Reconstructed)")
    # plt.ylabel("Amplitude")
    # plt.legend(loc='upper right')
    # plt.grid(True, alpha=0.3)

    # # 4. Error Signal
    # plt.subplot(4, 1, 4)
    # plt.plot(t, error_signal, color='purple', label='Error (True - Pred)')
    # plt.fill_between(t, error_signal, color='purple', alpha=0.2) # Shading
    # plt.title("Error Signal")
    # plt.xlabel("Time Samples")
    # plt.ylabel("Difference")
    # plt.legend(loc='upper right')
    # plt.grid(True, alpha=0.3)

    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()

def run_visualization():
    g = torch.Generator()
    g.manual_seed(SEED)
    # Load Models
    model_clip, model_converter = load_models()
    
    # Load Data
    try:
        test_dataset = LoadData(TEST_DATA_PATH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
        print(f"Loaded dataset from {TEST_DATA_PATH}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return


    seen_records = set()
    samples_collected = 0
    TARGET_SAMPLES = 10
    print(f"Running inference to find {TARGET_SAMPLES} unique records...")
    # Inference Loop
    print("Running inference...")
    with torch.no_grad():
        for i, (ecg, ppg, labels, groupID, record_names) in enumerate(test_loader):
            # Move to device and add channel dim
            ecg_target = ecg.to(DEVICE).float().unsqueeze(1)
            ppg_input = ppg.to(DEVICE).float().unsqueeze(1)

            # --- Forward Pass ---
            # 1. Get embedding from CLIP (we ignore logits here)
            ppg_embedding, feature_lists_PPG = model_clip(None, ppg_input)

            # 2. Decode embedding to ECG using Converter
            predicted_ecg = model_converter(ppg_embedding, feature_lists_PPG)

            # --- Visualization ---
            # Convert to CPU numpy for plotting
            ppg_np = ppg_input.cpu().squeeze().numpy()
            ecg_true_np = ecg_target.cpu().squeeze().numpy()
            ecg_pred_np = predicted_ecg.cpu().squeeze().numpy()
            current_batch_size = ppg_np.shape[0]
            for idx in range(current_batch_size):
                # Lấy tên record tương ứng
                current_rec_name = record_names[idx]

                # Kiểm tra xem record này đã vẽ chưa
                if current_rec_name not in seen_records:
                    print(f"Found unique record [{samples_collected + 1}/{TARGET_SAMPLES}]: {current_rec_name}")
                    
                    # Gọi hàm visualize
                    visualize_results(
                        ppg_np[idx], 
                        ecg_true_np[idx], 
                        ecg_pred_np[idx], 
                        samples_collected,  # Dùng biến đếm làm ID cho hình ảnh
                         current_rec_name
                    )
                    
                    # Đánh dấu đã xem và tăng biến đếm
                    seen_records.add(current_rec_name)
                    samples_collected += 1

                    # Điều kiện dừng: Đã đủ 4 mẫu khác nhau
                    if samples_collected >= TARGET_SAMPLES:
                        print("Done plotting 4 unique records.")
                        return


def run_loss():
    g = torch.Generator()
    g.manual_seed(SEED)
    # Load Models
    total_rmse = 0.0 
    total_pearson = 0.0
    total_prd = 0.0
    total_ssim = 0.0
    total_dtw = 0.0
    total_cosine = 0.0
    total_samples = 0
    model_clip, model_converter = load_models()
    
    # Load Data
    try:
        test_dataset = LoadData(TEST_DATA_PATH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
        print(f"Loaded dataset from {TEST_DATA_PATH}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return


    # Inference Loop
    print("Running...")
    with torch.no_grad():
        for i, (ecg, ppg, labels, groupID, record_names) in enumerate(test_loader):
            # Move to device and add channel dim
            ecg_target = ecg.to(DEVICE).float().unsqueeze(1)
            ppg_input = ppg.to(DEVICE).float().unsqueeze(1)

            # --- Forward Pass ---
            # 1. Get embedding from CLIP (we ignore logits here)
            ppg_embedding, feature_lists_PPG = model_clip(None, ppg_input)

            # 2. Decode embedding to ECG using Converter
            predicted_ecg = model_converter(ppg_embedding, feature_lists_PPG)

            # --- Visualization ---
            # Convert to CPU numpy for plotting
            ppg_np = ppg_input.cpu().squeeze().numpy()
            ecg_true_np = ecg_target.cpu().squeeze().numpy()
            ecg_pred_np = predicted_ecg.cpu().squeeze().numpy()
            current_batch_size = ecg_target.size(0)
            # rmse , pearson = calculate_metrics(ecg_true_np, ecg_pred_np)
            # total_rmse += rmse * current_batch_size
            # total_pearson += pearson * current_batch_size
            # total_prd += calculate_prd(ecg_true_np, ecg_pred_np) * current_batch_size
            # total_ssim += calculate_ssim_1d(ecg_true_np, ecg_pred_np) * current_batch_size
            # total_dtw += calculate_dtw_distance(ecg_true_np, ecg_pred_np) * current_batch_size
            total_cosine += calculate_cosine_similarity(ecg_true_np, ecg_pred_np) * current_batch_size
            total_samples += current_batch_size
            # print("shape: ", current_batch_size)
            # print(f"Record name {record_names}")
            # print(f"Batch {i+1} : RMSE = {rmse:.4f}, Pearson = {pearson:.4f}")
    # avg_rmse = total_rmse / total_samples
    # avg_pearson = total_pearson / total_samples
    # avg_prd = total_prd / total_samples
    # avg_ssim = total_ssim / total_samples
    # avg_dtw = total_dtw / total_samples
    avg_cosine = total_cosine / total_samples
    print(f"Average Cosine Similarity: {avg_cosine:.4f}")
    # print(f"Average : RMSE = {avg_rmse:.4f}, Pearson = {avg_pearson:.4f}, PRD = {avg_prd:.4f}, SSIM = {avg_ssim:.4f}, DTW = {avg_dtw:.4f}, Cosine = {avg_cosine:.4f}")
    return
if __name__ == "__main__":
    set_seed(SEED)
    run_visualization()
    run_loss()