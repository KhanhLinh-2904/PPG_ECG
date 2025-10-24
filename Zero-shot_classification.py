import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from CLIP import ECGEssembleCLIP
from load_data import LoadData
from load_data_ecg import LoadDataECG
from load_data_ppg import LoadDataPPG

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
OUTPUT_EMBED_DIM = 128 
INPUT_LENGTH = 2400 
EXAMPLE_LABELS = [
    "Normal Sinus Rhythm",
    "Atrial Fibrillation - AF"
]

def create_label_tokens(full_ppg_data, full_labels, device):
    """
    Tính toán tín hiệu PPG trung bình cho từng lớp (0 và 1) để tạo ra Label Tokens.
    
    Args:
        full_ppg_data (torch.Tensor): Tensor chứa tất cả tín hiệu PPG [N, 1, L].
        full_labels (torch.Tensor): Tensor chứa tất cả nhãn [N].
        device: Thiết bị (CPU/CUDA).
        
    Returns:
        torch.Tensor: Tensor chứa 2 Label Tokens (PPG trung bình), shape [2, L].
    """
    
    # Đảm bảo dữ liệu và nhãn trên CPU để dễ dàng thao tác với numpy
    ppg_np = full_ppg_data.squeeze(1).cpu().numpy() # shape [N, 2400]
    labels_np = full_labels.cpu().numpy()          # shape [N]
    
    print("\n--- Tính toán Label Tokens (Class Prototypes) ---")
    
    # 1. Tín hiệu trung bình cho lớp Normal (label 0)
    normal_indices = np.where(labels_np == 0)[0]
    if len(normal_indices) == 0:
        print("Cảnh báo: Không tìm thấy mẫu Normal (0). Sử dụng giá trị 0 ngẫu nhiên.")
        normal_mean_ppg = np.zeros(INPUT_LENGTH)
    else:
        normal_ppg_signals = ppg_np[normal_indices]
        normal_mean_ppg = np.mean(normal_ppg_signals, axis=0)
        print(f"Đã sử dụng {len(normal_indices)} mẫu cho Normal prototype.")

    # 2. Tín hiệu trung bình cho lớp AF (label 1)
    af_indices = np.where(labels_np == 1)[0]
    if len(af_indices) == 0:
        print("Cảnh báo: Không tìm thấy mẫu AF (1). Sử dụng giá trị 1 ngẫu nhiên.")
        af_mean_ppg = np.zeros(INPUT_LENGTH)
    else:
        af_ppg_signals = ppg_np[af_indices]
        af_mean_ppg = np.mean(af_ppg_signals, axis=0)
        print(f"Đã sử dụng {len(af_indices)} mẫu cho AF prototype.")
        
    # 3. Kết hợp thành Label Tokens
    # Tokens shape: [Num_Classes, INPUT_LENGTH]
    label_tokens_np = np.stack([normal_mean_ppg, af_mean_ppg])
    
    # Chuyển về Tensor và chuyển lên device (loại bỏ chiều channel = 1 vì encoder tự thêm)
    label_tokens = torch.from_numpy(label_tokens_np).float().to(device)
    
    return label_tokens

def zero_shot_evaluate(model, dataloader, device, label_tokens):
    """
    Thực hiện phân loại Zero-Shot cho tín hiệu ECG.
    Sử dụng Label Tokens (PPG Prototypes) để so sánh.
    """
    model.eval()
    
    # 1. Mã hóa các Nhãn (Label Prototypes)
    # label_tokens shape: [Num_Classes, L]. Thêm chiều channel = 1 để phù hợp với encoder.
    label_tokens_with_channel = label_tokens.unsqueeze(1) # [Num_Classes, 1, L]
    all_labels = []
    all_predictions = []
    print("\n--- Bắt đầu Phân loại Zero-Shot ---")
    
    with torch.no_grad():
        # dataloader trả về ecg, ppg (placeholder), và labels
        for ecg,_, labels in tqdm(dataloader, desc="Zero-Shot Testing", unit="batch"):
            # 2. Mã hóa ECG
            ecg = ecg.to(device).float().unsqueeze(1)
            # ppg = ppg.to(device).float().unsqueeze(1)

            # Dùng ECG Encoder để tạo vector nhúng cho tín hiệu ECG
            logits_per_ecg, logits_per_ppg = model(ecg, label_tokens_with_channel)
            predictions = torch.argmax(logits_per_ecg, dim=1)
            all_labels.append(labels)
            all_predictions.append(predictions)
   
        
    all_labels = torch.cat(all_labels).to(device) # [N_total]
    all_predictions = torch.cat(all_predictions).to(device) # [N_total]
    y_true = all_labels.cpu().numpy()
    y_pred = all_predictions.cpu().numpy()
    
    print(f"Tổng số mẫu: {len(y_true)}")
    
    # --- 4. TÍNH TOÁN CÁC METRICS ---
    
    # Accuracy (Độ chính xác)
    accuracy = accuracy_score(y_true, y_pred) * 100
    
    try:
        precision = precision_score(y_true, y_pred, average='binary', pos_label=1)
        recall = recall_score(y_true, y_pred, average='binary', pos_label=1)
        f1 = f1_score(y_true, y_pred, average='binary', pos_label=1)
    except ValueError as e:
        print(f"Cảnh báo: Không thể tính Precision/Recall/F1-score. Lỗi: {e}")
        # Xảy ra nếu lớp dương (label 1) không xuất hiện trong y_true hoặc y_pred
        precision, recall, f1 = 0.0, 0.0, 0.0
    
    # --- 5. IN KẾT QUẢ VÀ RETURN ---
    print("\n--- KẾT QUẢ METRICS ---")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    return accuracy, precision, recall, f1

if __name__ == '__main__':
    BATCH_SIZE = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")
    
    # 1. Tải toàn bộ dữ liệu PPG và nhãn (ví dụ: từ tập Validation/Test)
    # Trong môi trường thực, bạn sẽ tải dữ liệu thực tế tại đây.
    full_dataset = LoadData('datasets/MIMIC_test.npz')
    # Lấy toàn bộ dữ liệu để tính toán trung bình
    ppg_list = []
    labels_list = []
    # Sử dụng DataLoader để tải toàn bộ dữ liệu (cần thiết nếu dữ liệu lớn)
    temp_loader = DataLoader(full_dataset, batch_size=len(full_dataset), shuffle=False)
    
    # Load batch duy nhất
    for _, ppg, labels in temp_loader:
        ppg_list.append(ppg)
        labels_list.append(labels)
    
    full_ppg_data = torch.cat(ppg_list, dim=0) # [N, 1, 2400]
    full_labels = torch.cat(labels_list, dim=0) # [N]
    
    # 2. TẠO LABEL TOKENS DỰA TRÊN PPG TRUNG BÌNH
    # label_tokens sẽ có shape [2, 2400]
    # Khi được đưa vào Encoder, nó sẽ trở thành [2, 1, 2400] (do encoder tự thêm 1 channel)
    final_label_tokens = create_label_tokens(full_ppg_data, full_labels, device)

    print("\n========================================================")
    print(f"Đã tạo thành công Label Tokens:")
    print(f"Số lượng Tokens (Classes): {final_label_tokens.shape[0]}")
    print(f"Chiều dài Tín hiệu: {final_label_tokens.shape[1]}")
    print(f"Shape của Tensor Label Tokens: {final_label_tokens.shape}")
    print("========================================================")
    
    # Ghi chú: Tensor 'final_label_tokens' này chính là đầu vào cho hàm zero_shot_evaluate
    # Ví dụ: zero_shot_accuracy = zero_shot_evaluate(model, test_loader, device, final_label_tokens)
    model = ECGEssembleCLIP(embed_dim=OUTPUT_EMBED_DIM).to(device)
    try:
        model.load_state_dict(torch.load('ecg_ppg_clip_best_model.pth', map_location=device))
        print("\n=> Tải trọng số mô hình từ 'ecg_ppg_clip_best_model.pth' thành công.")
    except FileNotFoundError:
        print("\n=> Lỗi: Không tìm thấy 'ecg_ppg_clip_best_model.pth'. Chạy đánh giá Zero-Shot với mô hình chưa huấn luyện (kết quả sẽ ngẫu nhiên).")

    # Tải dữ liệu kiểm tra Zero-Shot (chứa ECG và nhãn thực tế)
    test_dataset = LoadData('datasets/MIMIC_test.npz')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 4. CHẠY ĐÁNH GIÁ ZERO-SHOT
    zero_shot_accuracy,  precision, recall, f1 = zero_shot_evaluate(model, test_loader, device, final_label_tokens)
    
    print(f"\n========================================================")
    print(f" KẾT QUẢ ĐÁNH GIÁ ZERO-SHOT (Normal vs AF)")
    print(f" Accuracy: {zero_shot_accuracy:.2f}%")
    print(f"========================================================")

    