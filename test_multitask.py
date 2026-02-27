import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from load_data import LoadData, PPG2ECG_Dataset
from CLIP import ECGDecoder_UNet, ECGEssembleCLIP
import random
from scipy import signal
import os

from metric import calculate_cosine_similarity, calculate_dtw_distance, calculate_metrics

# --- CONFIGURATION ---
SEED = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16 
INPUT_LENGTH = 2048
OUTPUT_EMBED_DIM = 128
TEST_DATA_PATH = 'datasets/total_z_test.npz' 
CLIP_MODEL_PATH = "multitask_clip_best_model.pth"
DECODER_MODEL_PATH = "multitask_decoder_best_model.pth"
ppg_sqi_thresh = 0.3
ecg_sqi_thresh = 0.8
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_models():
    print(f"Loading models on {DEVICE}...")
    model_clip = ECGEssembleCLIP(embed_dim=OUTPUT_EMBED_DIM).to(DEVICE)
    model_converter = ECGDecoder_UNet(bottleneck_channels=2048).to(DEVICE)

    if torch.cuda.is_available():
        clip_weights = torch.load(CLIP_MODEL_PATH)
        decoder_weights = torch.load(DECODER_MODEL_PATH)
    else:
        clip_weights = torch.load(CLIP_MODEL_PATH, map_location='cpu')
        decoder_weights = torch.load(DECODER_MODEL_PATH, map_location='cpu')

    model_clip.load_state_dict(clip_weights)
    model_converter.load_state_dict(decoder_weights)
    model_clip.eval()
    model_converter.eval()
    return model_clip, model_converter

def visualize_results(ppg, ecg_true, ecg_pred, sample_idx, record_name):
    """ Vẽ biểu đồ với kích thước cửa sổ bình thường """
    print(f"Visualizing Record: {record_name}")
    t = np.arange(len(ppg))

    # Kích thước figure được giữ ở mức hợp lý (12x10) thay vì ép toàn màn hình
    plt.figure(figsize=(12, 10))
    
    plt.suptitle(f"Record: {record_name} - Sample ID: {sample_idx}", fontsize=14, fontweight='bold')

    # 1. PPG Input
    plt.subplot(4, 1, 1)
    plt.plot(t, ppg, color='green', label='Input PPG')
    plt.title("Input PPG Signal")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    # 2. Ground Truth ECG
    plt.subplot(4, 1, 2)
    plt.plot(t, ecg_true, color='blue', label='Ground Truth ECG')
    plt.title("Ground Truth ECG")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    # 3. Predicted ECG
    plt.subplot(4, 1, 3)
    plt.plot(t, ecg_pred, color='red', label='Predicted ECG')
    plt.title("Predicted ECG (Reconstructed)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    # 4. Comparison
    plt.subplot(4, 1, 4)
    plt.plot(t, ecg_true, color='black', label='Ground Truth', alpha=0.7)
    plt.plot(t, ecg_pred, color='red', label='Predicted', linestyle='--', alpha=0.8)
    plt.title("Comparison: Ground Truth vs Predicted ECG")
    plt.xlabel("Time Samples")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def save_ecg_reconstruction(output_path = "AF_Detection/ecg_reconstructions.npz"):
    set_seed(SEED)
    model_clip, model_converter = load_models()
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
   
    print("Running inference for visualization...")
    with torch.no_grad():
        for i, (ecg, ppg, record_names, label) in enumerate(test_loader):
            ppg_input = ppg.to(DEVICE).float().unsqueeze(1)
            print("record name: ", record_names)
            ppg_embedding, feature_lists_PPG = model_clip(None, ppg_input)
            predicted_ecg = model_converter(ppg_embedding, feature_lists_PPG)
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

def run_visualization():
    set_seed(SEED)
    model_clip, model_converter = load_models()
    
    # Ép mô hình về chế độ test (rất quan trọng để cố định BatchNorm và Dropout)
    model_clip.eval()
    model_converter.eval()
    
    seen_records = set()

    try:
        test_dataset = PPG2ECG_Dataset(TEST_DATA_PATH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    except Exception as e:
        print(f"Error: {e}")
        return
   
    print("Running inference for visualization on ALL valid samples...")
    with torch.no_grad():
        for batch in test_loader:
            ecg, ppg, record_names, ppg_sqi, ecg_sqi = batch
            
            # 1. TẠO MẶT NẠ LỌC CHẤT LƯỢNG (BOOLEAN MASK)
            valid_mask = (ppg_sqi >= ppg_sqi_thresh) & (ecg_sqi >= ecg_sqi_thresh)
            num_valid = valid_mask.sum().item()
            
            # Nếu cả batch đều là rác thì bỏ qua luôn, không đưa vào GPU
            if num_valid == 0:
                continue
                
            # 2. TRÍCH XUẤT DỮ LIỆU SẠCH
            valid_ppg = ppg[valid_mask].to(DEVICE).float().unsqueeze(1)
            valid_ecg = ecg[valid_mask] # Giữ Groundtruth ở CPU/RAM
            
            # Lọc danh sách tên record tương ứng với các mẫu sạch
            valid_records = [record_names[j] for j in range(len(record_names)) if valid_mask[j]]

            # 3. CHẠY SUY LUẬN CHỈ TRÊN DỮ LIỆU SẠCH
            ppg_embedding, feature_lists_PPG = model_clip(None, valid_ppg)
            predicted_ecg = model_converter(ppg_embedding, feature_lists_PPG)

            # 4. CHUYỂN ĐỔI SANG NUMPY ĐỂ VẼ HÌNH
            # Dùng squeeze(1) để lột bỏ chiều Channel, atleast_2d để phòng hờ batch size = 1
            ppg_np = np.atleast_2d(valid_ppg.cpu().squeeze(1).numpy())
            ecg_true_np = np.atleast_2d(valid_ecg.numpy())
            ecg_pred_np = np.atleast_2d(predicted_ecg.cpu().squeeze(1).numpy())

            # 5. VẼ HÌNH TẤT CẢ CÁC MẪU ĐẠT CHUẨN
            for idx in range(num_valid):
                current_rec_name = valid_records[idx]
                
                # Chỉ vẽ nếu record này chưa từng được vẽ trước đó
                if current_rec_name not in seen_records:
                    visualize_results(
                        ppg_np[idx], 
                        ecg_true_np[idx], 
                        ecg_pred_np[idx], 
                        len(seen_records), 
                        current_rec_name
                    )
                    seen_records.add(current_rec_name)
                    
    print(f"\nĐã hoàn tất quá trình vẽ biểu đồ. Tổng số mẫu đạt chuẩn SQI được vẽ: {len(seen_records)}")

def run_loss():
    model_clip, model_converter = load_models()
    
    # Chuyển mô hình sang chế độ đánh giá
    model_clip.eval()
    model_converter.eval()
    
    test_dataset = PPG2ECG_Dataset(TEST_DATA_PATH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    total_rmse, total_pearson, total_dtw, total_cosine = 0, 0, 0, 0
    total_initial_samples = 0
    total_valid_samples = 0

    print("Calculating Metrics on Valid Samples...")
    
    with torch.no_grad():
        for batch in test_loader:
            # 1. Giải nén dữ liệu từ DataLoader
            ecg, ppg, record_names, ppg_sqi, ecg_sqi = batch
            batch_size = ecg.size(0)
            total_initial_samples += batch_size
            
            # 2. TẠO MẶT NẠ LỌC (BOOLEAN MASK)
            valid_mask = (ppg_sqi >= ppg_sqi_thresh) & (ecg_sqi >= ecg_sqi_thresh)
            num_valid = valid_mask.sum().item()
            
            if num_valid == 0:
                continue # Bỏ qua batch nếu toàn bộ là rác
                
            total_valid_samples += num_valid
            
            # 3. TRÍCH XUẤT DỮ LIỆU SẠCH
            # Chỉ đẩy những mẫu hợp lệ lên GPU để tiết kiệm RAM
            valid_ppg = ppg[valid_mask].to(DEVICE).float().unsqueeze(1)
            valid_ecg_np = ecg[valid_mask].numpy() # Giữ ECG ở CPU dạng Numpy để lát tính metric
            
            # Lọc tên record tương ứng
            valid_records = [record_names[j] for j in range(batch_size) if valid_mask[j]]

            # 4. CHẠY SUY LUẬN (INFERENCE)
            # Giả định model_clip không cần nhãn ecg ở bước inference nên truyền None
            ppg_embedding, feature_lists_PPG = model_clip(None, valid_ppg)
            predicted_ecg = model_converter(ppg_embedding, feature_lists_PPG)

            # Đưa kết quả dự đoán về CPU và Numpy
            pred_ecg_np = predicted_ecg.cpu().squeeze(1).numpy()

            # Đảm bảo mảng luôn là 2D (phòng trường hợp batch sau khi lọc chỉ còn đúng 1 mẫu)
            valid_ecg_np = np.atleast_2d(valid_ecg_np)
            pred_ecg_np = np.atleast_2d(pred_ecg_np)

            # 5. TÍNH TOÁN METRIC (Chỉ trên dữ liệu sạch)
            for b in range(num_valid):
                true_s = valid_ecg_np[b]
                pred_s = pred_ecg_np[b]

                rmse, pearson = calculate_metrics(true_s, pred_s)
                dtw = calculate_dtw_distance(true_s, pred_s)
                cosine = calculate_cosine_similarity(true_s, pred_s)
                
                # In chi tiết nếu cần thiết
                # print(f"Record: {valid_records[b]} | rRMSE: {rmse:.4f}, Pearson: {pearson:.4f}, DTW: {dtw:.4f}, Cosine: {cosine:.4f}")
                
                total_rmse += rmse
                total_pearson += pearson
                total_dtw += dtw
                total_cosine += cosine

    # 6. BÁO CÁO KẾT QUẢ TỔNG QUAN
    print("\n" + "="*40)
    print("FINAL TESTING RESULTS")
    print("="*40)
    print(f"Total samples scanned  : {total_initial_samples}")
    print(f"Valid samples evaluated: {total_valid_samples} ({(total_valid_samples/total_initial_samples)*100:.2f}%)")
    print("-" * 40)
    
    if total_valid_samples > 0:
        print(f"Average rRMSE  : {total_rmse / total_valid_samples:.4f}")
        print(f"Average Pearson: {total_pearson / total_valid_samples:.4f}")
        print(f"Average DTW    : {total_dtw / total_valid_samples:.4f}")
        print(f"Average Cosine : {total_cosine / total_valid_samples:.4f}")
    else:
        print("CẢNH BÁO: Không có mẫu nào vượt qua được ngưỡng SQI đã đặt!")
    print("="*40)

if __name__ == "__main__":
    # run_visualization()
    run_loss()
    # save_ecg_reconstruction("AF_Detection/total_test_ecg_reconstructions_no_mm.npz")