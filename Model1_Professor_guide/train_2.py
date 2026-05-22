import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# Import hàm LoadData từ file của bạn
try:
    from load_data import LoadData
except ImportError:
    print("Lỗi: Không tìm thấy file load_data.py trong thư mục hiện tại!")
    exit()

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN VÀ BATCH SIZE
# ==========================================
TRAIN_PATH = '/home/linhhima/Diffusion_datasets/combined_segment_split_test.npz'
VAL_PATH = '/home/linhhima/Diffusion_datasets/combined_segment_split_val.npz'

# Để batch_size lớn (ví dụ 256 hoặc 512) để quá trình quét diễn ra nhanh nhất có thể
BATCH_SIZE = 256  
NUM_WORKERS = 4

def scan_dataset(dataloader, dataset_name):
    print(f"\n{'='*50}")
    print(f"[*] BẮT ĐẦU QUÉT: {dataset_name.upper()}")
    print(f"{'='*50}")
    
    total_nan_ecg = 0
    total_inf_ecg = 0
    total_nan_ppg = 0
    total_inf_ppg = 0
    
    # Quét qua từng batch
    for batch_idx, (ecg, ppg, _) in enumerate(tqdm(dataloader, desc=f"Scanning", unit="batch")):
        # Kiểm tra ECG
        if torch.isnan(ecg).any():
            total_nan_ecg += 1
            print(f"\n -> [LỖI] Phát hiện NaN trong ECG tại batch {batch_idx}")
        if torch.isinf(ecg).any():
            total_inf_ecg += 1
            print(f"\n -> [LỖI] Phát hiện Inf (Infinity) trong ECG tại batch {batch_idx}")
            
        # Kiểm tra PPG
        if torch.isnan(ppg).any():
            total_nan_ppg += 1
            print(f"\n -> [LỖI] Phát hiện NaN trong PPG tại batch {batch_idx}")
        if torch.isinf(ppg).any():
            total_inf_ppg += 1
            print(f"\n -> [LỖI] Phát hiện Inf (Infinity) trong PPG tại batch {batch_idx}")

    # Tổng kết
    print("\n--- BÁO CÁO TỔNG KẾT ---")
    if (total_nan_ecg + total_inf_ecg + total_nan_ppg + total_inf_ppg) == 0:
        print(f"✅ [OK] Tập dữ liệu {dataset_name} HOÀN TOÀN SẠCH (Không có NaN / Inf).")
        return True
    else:
        print(f"❌ [CẢNH BÁO] Tập dữ liệu {dataset_name} BỊ LỖI!")
        print(f"  - Số batch chứa ECG bị NaN: {total_nan_ecg}")
        print(f"  - Số batch chứa ECG bị Inf: {total_inf_ecg}")
        print(f"  - Số batch chứa PPG bị NaN: {total_nan_ppg}")
        print(f"  - Số batch chứa PPG bị Inf: {total_inf_ppg}")
        return False

if __name__ == "__main__":
    # 1. Kiểm tra xem file có tồn tại không
    if not os.path.exists(TRAIN_PATH):
        print(f"Không tìm thấy file: {TRAIN_PATH}")
    else:
        # Load và kiểm tra tập Train
        train_loader = DataLoader(LoadData(TRAIN_PATH), batch_size=BATCH_SIZE, 
                                  shuffle=False, num_workers=NUM_WORKERS)
        scan_dataset(train_loader, "Tập Train (combined_train.npz)")

    # 2. Kiểm tra tập Validation
    if not os.path.exists(VAL_PATH):
        print(f"\nKhông tìm thấy file: {VAL_PATH}")
    else:
        # Load và kiểm tra tập Validation
        val_loader = DataLoader(LoadData(VAL_PATH), batch_size=BATCH_SIZE, 
                                shuffle=False, num_workers=NUM_WORKERS)
        scan_dataset(val_loader, "Tập Validation (combined_val.npz)")
        
    print("\n[*] Quá trình kiểm tra đã hoàn tất!")