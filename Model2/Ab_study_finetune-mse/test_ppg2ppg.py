import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy.signal import correlate

# Import các module PPG tương ứng từ file chứa kiến trúc mô hình của bạn
# Giả sử file mô hình của bạn tên là ppg2ppg.py
from ppg2ppg import PPGAutoencoder, PPGAEConfig 
from metric import calculate_cosine_similarity, calculate_dtw_distance, calculate_metrics

# ==========================================
# --- CONFIGURATION ---
# ==========================================
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
SEQ_LENGTH = 2400

# Đường dẫn dữ liệu test và trọng số mô hình PPG AE
TEST_DATA_PATH = '/home/linhhima/Diffusion_datasets/combined_segment_split_test.npz'
MODEL_PATH = '/home/linhhima/PPG_ECG/Model2/Ab_study/saved_models_ppg_ae_segment/best_ppg_autoencoder.pth'

# ==========================================

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_model():
    print(f"[*] Đang tải mô hình PPG Autoencoder lên {DEVICE}...")
    
    # Khởi tạo cấu hình cho mô hình PPG AE khớp với file thiết kế mô hình
    cfg = PPGAEConfig(
        input_length=SEQ_LENGTH,
        ppg_in_channels=1,
        dims=(64, 128, 256, 512),
        depths=(2, 2, 4, 2),
        latent_channels=16,
        latent_length=75,
        attn_heads=8,
        attn_dropout=0.0,
        use_derivatives=True,
        global_latent_dim=128,
        trend_poly=2,
    )
    
    model = PPGAutoencoder(cfg).to(DEVICE)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"[!] Không tìm thấy file trọng số tại: {MODEL_PATH}")

    # Tải trọng số (hỗ trợ checkpoint lưu dạng dictionary hoặc chỉ lưu state_dict thông thường)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model

def visualize_results(ppg_true, ppg_pred, record_name):
    print(f"Visualizing Record: {record_name}")
    t_ppg = np.arange(len(ppg_true))

    plt.figure(figsize=(12, 8))
    plt.suptitle(f"PPG Autoencoder Reconstruction | Record: {record_name}", fontsize=14, fontweight='bold')

    plt.subplot(3, 1, 1)
    plt.plot(t_ppg, ppg_true, color='green', label='Input PPG (Ground Truth)')
    plt.title("Input PPG Signal")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    plt.subplot(3, 1, 2)
    plt.plot(t_ppg, ppg_pred, color='red', label='Reconstructed PPG')
    plt.title("Reconstructed PPG Signal")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    plt.subplot(3, 1, 3)
    plt.plot(t_ppg, ppg_true, color='black', label='Ground Truth', alpha=0.7)
    plt.plot(t_ppg, ppg_pred, color='red', label='Reconstructed', linestyle='--', alpha=0.8)
    plt.title("Comparison: Ground Truth vs Reconstructed PPG")
    plt.xlabel("Time Samples")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Hiển thị trực tiếp đồ thị lên màn hình
    plt.show()

def run_visualization():
    set_seed(SEED)
    model = load_model()
    seen_records = set()

    try:
        from load_data import LoadData
        test_dataset = LoadData(TEST_DATA_PATH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    except Exception as e:
        print(f"[!] Lỗi khi load data: {e}")
        return
   
    print("\n[*] Đang chạy suy luận (Inference) để trực quan hóa tín hiệu PPG...")
    with torch.no_grad():
        for batch_data in test_loader:
            # 💡 Lưu ý từ Code Linh: Dựa vào cấu trúc cũ, batch_data[0] là ECG, batch_data[1] là PPG.
            # Vì đây là bài toán PPG-to-PPG nên ta lấy dữ liệu PPG tại index 1 để đưa vào mô hình.
            ppg = batch_data[1]
            record_names = batch_data[2]  # Giả sử index 2 lưu giữ tên bản ghi (record names)

            # Định dạng lại shape đảm bảo cấu trúc 1D CNN [B, 1, 2400]
            ppg_input = ppg.float().unsqueeze(1).to(DEVICE) if ppg.dim() == 2 else ppg.float().to(DEVICE)

            # Mô hình PPGAutoencoder trả về dạng Dictionary
            outputs = model(ppg_input)
            predicted_ppg = outputs["reconstructed_ppg"]

            ppg_true_np = np.atleast_2d(ppg.cpu().squeeze().numpy())
            ppg_pred_np = np.atleast_2d(predicted_ppg.cpu().squeeze().numpy())

            for idx in range(ppg_true_np.shape[0]):
                current_rec_name = record_names[idx]
                
                if current_rec_name not in seen_records:
                    visualize_results(
                        ppg_true_np[idx], 
                        ppg_pred_np[idx], 
                        current_rec_name
                    )
                    seen_records.add(current_rec_name)
                
                # Chỉ hiển thị demo cấu trúc 5 samples độc lập rồi dừng lại
                if len(seen_records) >= 5:
                    print("[*] Đã hiển thị xong 5 samples demo trực quan PPG.")
                    return

def align_signals(true_s: np.ndarray, pred_s: np.ndarray) -> np.ndarray:
    """ Dùng Cross-Correlation để tìm độ lệch pha và dịch chuyển tín hiệu dự đoán về khớp với gốc. """
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
    set_seed(SEED)
    model = load_model()
    
    try:
        from load_data import LoadData
        test_dataset = LoadData(TEST_DATA_PATH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    except Exception as e:
        print(f"[!] Lỗi khi load data: {e}")
        return

    metrics = {
        'before': {'rmse': 0, 'pearson': 0, 'dtw': 0, 'cosine': 0},
        'after':  {'rmse': 0, 'pearson': 0, 'dtw': 0, 'cosine': 0}
    }
    total_samples = 0

    print("\n[*] Đang tính toán các chỉ số lỗi tín hiệu PPG (Metrics)...")
    with torch.no_grad():
        for batch_data in test_loader:
            # Lấy tín hiệu PPG đầu vào tương tự như nhánh suy luận trên
            ppg = batch_data[1]

            ppg_input = ppg.float().unsqueeze(1).to(DEVICE) if ppg.dim() == 2 else ppg.float().to(DEVICE)

            outputs = model(ppg_input)
            predicted_ppg = outputs["reconstructed_ppg"]

            ppg_true_np = np.atleast_2d(ppg.cpu().squeeze().numpy())
            ppg_pred_np = np.atleast_2d(predicted_ppg.cpu().squeeze().numpy())

            for b in range(ppg_true_np.shape[0]):
                true_s = ppg_true_np[b]
                pred_s = ppg_pred_np[b]

                # 1. TRƯỚC KHI ĐỒNG BỘ PHA (BEFORE ALIGN)
                rmse_b, pearson_b = calculate_metrics(true_s, pred_s)
                dtw_b = calculate_dtw_distance(true_s, pred_s)
                cosine_b = calculate_cosine_similarity(true_s, pred_s)
                
                metrics['before']['rmse'] += rmse_b
                metrics['before']['pearson'] += pearson_b
                metrics['before']['dtw'] += dtw_b
                metrics['before']['cosine'] += cosine_b

                # 2. DỊCH CHUYỂN ĐỒNG BỘ PHA (SHIFT ALIGNMENT)
                aligned_pred_s = align_signals(true_s, pred_s)

                # 3. SAU KHI ĐỒNG BỘ PHA (AFTER ALIGN)
                rmse_a, pearson_a = calculate_metrics(true_s, aligned_pred_s)
                dtw_a = calculate_dtw_distance(true_s, aligned_pred_s)
                cosine_a = calculate_cosine_similarity(true_s, aligned_pred_s)

                metrics['after']['rmse'] += rmse_a
                metrics['after']['pearson'] += pearson_a
                metrics['after']['dtw'] += dtw_a
                metrics['after']['cosine'] += cosine_a
                
                total_samples += 1

    # In bảng kết quả tổng hợp
    print(f"\n{'='*48}")
    print(f"KẾT QUẢ ĐÁNH GIÁ PPG AUTOENCODER ({total_samples} samples)")
    print(f"{'='*48}")
    print(f"{'Metric':<12} | {'Trước khi Align':<15} | {'Sau khi Align':<15}")
    print(f"{'-'*48}")
    print(f"RMSE         | {metrics['before']['rmse']/total_samples:<15.4f} | {metrics['after']['rmse']/total_samples:<15.4f}")
    print(f"Pearson      | {metrics['before']['pearson']/total_samples:<15.4f} | {metrics['after']['pearson']/total_samples:<15.4f}")
    print(f"DTW          | {metrics['before']['dtw']/total_samples:<15.4f} | {metrics['after']['dtw']/total_samples:<15.4f}")
    print(f"Cosine       | {metrics['before']['cosine']/total_samples:<15.4f} | {metrics['after']['cosine']/total_samples:<15.4f}")
    print(f"{'='*48}")


if __name__ == "__main__":
    
    # run_visualization()
    run_loss()