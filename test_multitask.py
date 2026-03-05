import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from load_data import LoadData, PPG2ECG_Dataset, quality_aware_collate_fn, PPG2ECG_BUT_Dataset
from CLIP import ECGDecoder_UNet, ECGEssembleCLIP
import random
from scipy import signal
import os
from tqdm import tqdm
import torch.nn.functional as F
from metric import calculate_cosine_similarity, calculate_dtw_distance, calculate_metrics
from train_multitask import TemporalNoiseFilter

# --- CONFIGURATION ---
SEED = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16 
INPUT_LENGTH = 2048
OUTPUT_EMBED_DIM = 128
TEST_DATA_PATH = 'processed_data/mimic3_v1_test.npz' 
CLIP_MODEL_PATH = "multitask_clip_best_model.pth"
DECODER_MODEL_PATH = "multitask_decoder_best_model.pth"
NOISE_FILTER_MODEL_PATH = "multitask_noise_filter_best.pth"
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
    
    # 1. Khởi tạo CẢ 3 MÔ HÌNH và đưa lên thiết bị (GPU/CPU)
    noise_filter = TemporalNoiseFilter(in_channels=1).to(DEVICE)
    model_clip = ECGEssembleCLIP(embed_dim=OUTPUT_EMBED_DIM).to(DEVICE)
    model_converter = ECGDecoder_UNet(bottleneck_channels=2048, target_length=2400).to(DEVICE)

    # 2. Tải file trọng số (weights) một cách an toàn
    if torch.cuda.is_available():
        noise_weights = torch.load(NOISE_FILTER_MODEL_PATH)
        clip_weights = torch.load(CLIP_MODEL_PATH)
        decoder_weights = torch.load(DECODER_MODEL_PATH)
    else:
        noise_weights = torch.load(NOISE_FILTER_MODEL_PATH, map_location='cpu')
        clip_weights = torch.load(CLIP_MODEL_PATH, map_location='cpu')
        decoder_weights = torch.load(DECODER_MODEL_PATH, map_location='cpu')

    # 3. Nạp trọng số vào cấu trúc mạng Nơ-ron
    noise_filter.load_state_dict(noise_weights)
    model_clip.load_state_dict(clip_weights)
    model_converter.load_state_dict(decoder_weights)
    
    # 4. Ép TẤT CẢ về chế độ đánh giá (Cực kỳ quan trọng để cố định BatchNorm)
    noise_filter.eval()
    model_clip.eval()
    model_converter.eval()
    
    # 5. Trả về đúng 3 giá trị theo thứ tự mà run_loss() đang chờ
    return noise_filter, model_clip, model_converter

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
    
    # 1. Tải mô hình và chuyển sang chế độ Eval
    noise_filter, model_clip, model_converter = load_models()
    noise_filter.eval()
    model_clip.eval()
    model_converter.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        test_dataset = PPG2ECG_Dataset(TEST_DATA_PATH)
        total_test_samples = len(test_dataset)
        print(f"Tổng số mẫu trong tập kiểm tra: {total_test_samples}")
        # Để trực quan hóa, thường ta dùng Batch Size nhỏ hoặc lấy mẫu ngẫu nhiên
        test_loader = DataLoader(test_dataset, batch_size=4, 
                                 shuffle=False, num_workers=4,
                                 pin_memory=True, collate_fn=quality_aware_collate_fn)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
   
    print("🚀 Running inference for visualization...")
    
    with torch.no_grad():
        for batch in test_loader:
            ecg, ppg, record_names, ppg_mask, ppg_sqi, ecg_sqi = batch
            
            # Chuẩn bị dữ liệu đầu vào
            ppg_input = ppg.to(device).float().unsqueeze(1)
            ecg_target = ecg.to(device).float().unsqueeze(1)
            true_mask = ppg_mask.to(device).float().unsqueeze(1)

            # 2. Chạy qua hệ thống model
            # Bước 1: Lọc nhiễu PPG
            masked_ppg, pred_mask, _ = noise_filter(ppg_input)
            
            # Bước 2: Trích xuất embedding và feature maps
            ppg_embedding, feature_lists_PPG = model_clip(None, masked_ppg)
            
            # Bước 3: Tái tạo ECG từ PPG đã lọc
            predicted_ecg = model_converter(ppg_embedding, feature_lists_PPG)

            # 3. Vẽ biểu đồ so sánh cho từng mẫu trong Batch
            for i in range(ppg_input.size(0)):
                fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
                
                # Tín hiệu PPG đầu vào và Mask dự đoán
                axes[0].plot(ppg[i].cpu().numpy(), color='blue', label='Original PPG')
                axes[0].set_title(f"Record: {record_names[i]} | PPG SQI: {ppg_sqi[i]:.2f}")
                axes[0].legend(loc='upper right')

                # Mask (Chất lượng tín hiệu) - So sánh Ground Truth và Prediction
                axes[1].fill_between(range(len(ppg_mask[i])), ppg_mask[i].cpu().numpy(), color='green', alpha=0.3, label='GT Mask')
                axes[1].plot(pred_mask[i, 0].cpu().numpy(), color='red', linestyle='--', label='Pred Mask')
                axes[1].set_title("Signal Quality Mask (Noise Detection)")
                axes[1].legend(loc='upper right')

                # ECG thật (Ground Truth)
                axes[2].plot(ecg[i].cpu().numpy(), color='black', label='Ground Truth ECG')
                axes[2].set_title(f"Target ECG | ECG SQI: {ecg_sqi[i]:.2f}")
                axes[2].legend(loc='upper right')

                # ECG dự đoán (Reconstructed)
                axes[3].plot(predicted_ecg[i, 0].cpu().numpy(), color='crimson', label='Reconstructed ECG')
                axes[3].set_title("Model Predicted ECG")
                axes[3].set_xlabel("Samples")
                axes[3].legend(loc='upper right')

                plt.tight_layout()
                plt.show()


def run_loss():
    # 1. Tải mô hình và chuyển sang chế độ đánh giá
    noise_filter, model_clip, model_converter = load_models()
    noise_filter.eval()
    model_clip.eval()
    model_converter.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Khởi tạo Dataset và DataLoader
    test_dataset = PPG2ECG_Dataset(TEST_DATA_PATH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                             shuffle=False, num_workers=4,
                             pin_memory=True, collate_fn=quality_aware_collate_fn)

    # Khởi tạo các biến tích lũy
    total_rmse, total_pearson, total_dtw, total_cosine = 0, 0, 0, 0
    total_samples = 0

    print(f"🔍 Đang tính toán Metrics trên TOÀN BỘ {len(test_dataset)} mẫu tín hiệu...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Đang đánh giá"):
            # Lấy dữ liệu từ batch
            ecg, ppg, record_names, ppg_mask, ppg_sqi, ecg_sqi = batch
            
            # Đưa lên thiết bị tính toán
            ppg_input = ppg.to(device).float().unsqueeze(1)    # [Batch, 1, 300]
            ecg_target = ecg.to(device).float().unsqueeze(1)   # [Batch, 1, 10000]

            # 2. Quy trình suy luận (Inference)
            # Bước A: Qua bộ lọc nhiễu (Dù không dùng mask để lọc metric, vẫn cần qua filter để làm sạch đầu vào)
            masked_ppg, _, _ = noise_filter(ppg_input)
            
            # Bước B: Trích xuất embedding từ PPG
            ppg_embedding, feature_lists_PPG = model_clip(None, masked_ppg)
            
            # Bước C: Tái tạo tín hiệu ECG
            predicted_ecg = model_converter(ppg_embedding, feature_lists_PPG)

            # 3. Đồng bộ kích thước dự đoán lên 10.000 mẫu để khớp với Ground Truth
            if predicted_ecg.shape[-1] != ecg_target.shape[-1]:
                predicted_ecg = F.interpolate(predicted_ecg, size=ecg_target.shape[-1], mode='linear')

            # 4. Tính toán Metrics cho từng mẫu trong Batch (Sử dụng toàn bộ 10.000 điểm)
            for i in range(ecg_target.size(0)):
                # Lấy tín hiệu 1D
                s_true = ecg_target[i, 0]
                s_pred = predicted_ecg[i, 0]
                
                # Gọi các hàm bắt buộc theo yêu cầu
                rmse, pearson = calculate_metrics(s_true, s_pred)
                dtw = calculate_dtw_distance(s_true, s_pred)
                cosine = calculate_cosine_similarity(s_true, s_pred)
                
                # Cộng dồn kết quả
                total_rmse += rmse
                total_pearson += pearson
                total_dtw += dtw
                total_cosine += cosine
                total_samples += 1

    # 5. Xuất báo cáo kết quả trung bình
    if total_samples > 0:
        print("\n" + "="*40)
        print("📊 BÁO CÁO HIỆU SUẤT TRÊN TOÀN BỘ TẬP TEST")
        print(f"Tổng số mẫu: {total_samples}")
        print(f"RMSE trung bình:    {total_rmse / total_samples:.4f}")
        print(f"Pearson trung bình: {total_pearson / total_samples:.4f}")
        print(f"DTW trung bình:     {total_dtw / total_samples:.4f}")
        print(f"Cosine trung bình:  {total_cosine / total_samples:.4f}")
        print("="*40)
    else:
        print("⚠️ Không có dữ liệu để đánh giá.")
    
   

if __name__ == "__main__":
    run_visualization()
    # run_loss()
    # save_ecg_reconstruction("AF_Detection/total_test_ecg_reconstructions_no_mm.npz")