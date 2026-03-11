
import numpy as np
import os
from preprocessing import SignalProcessor
from utils import calculate_sq_mask
import matplotlib.pyplot as plt
# --- CẤU HÌNH ---
PATH = "/home/linhhima/Pre_processing_data/Datasets/deepbeat_data_extract/"
FILES = ["train_cleaned_signals.npz", "val_cleaned_signals.npz", "test_cleaned_signals.npz"]
SAVE_PATH = os.path.join("processed_data", "deepbeat_combined.npz")
FS = 32

# Khởi tạo bộ tiền xử lý
signal_processor = SignalProcessor(fs=FS)

def mask_filter(ppg): 
    sq_mask_array = calculate_sq_mask(ppg, fs=FS)
    ppg_sqi = np.mean(sq_mask_array)
    return ppg_sqi < 0.3

def combine_and_preprocess_datasets():
    all_ppgs = []
    all_labels = []
    global_stats = {
        'total_raw': 0,
        'nan_flat_removed': 0,
        'nan_flat_af_removed': 0,
        'quality_removed': 0,
        'quality_af_removed': 0,
        'final_count': 0
    }
    visualized_af = False
    visualized_non_af = False
    for file_name in FILES:
        full_path = os.path.join(PATH, file_name)
        if not os.path.exists(full_path):
            print(f"Not finding file: {file_name}")
            continue
            
        print(f" {file_name}...")
        data = np.load(full_path, allow_pickle=True)
        
        x_data = data['x']
        y_data = data['y']
        # print("y_data:", y_data)
        global_stats['total_raw'] += len(x_data)
        for i in range(len(x_data)):
            segment = x_data[i].flatten()
            label_val = y_data[i]
            is_af = (label_val == 1)
            try:
                if any(np.isnan(segment)) or np.std(segment) < 1e-5:
                    global_stats['nan_flat_removed'] += 1
                    if is_af: global_stats['nan_flat_af_removed'] += 1
                    continue
                    
                processed_segment = signal_processor.preprocessing_PPG(segment)
                if mask_filter(processed_segment):
                    # count_filtered += 1
                    global_stats['quality_removed'] += 1
                    if is_af: global_stats['quality_af_removed'] += 1
                    continue

                all_ppgs.append(processed_segment)
                label_val = y_data[i]
                all_labels.append(label_val)

                # # 3. Logic hiển thị mẫu đại diện (Chỉ chạy 1 lần cho mỗi loại nhãn)
                # is_af = (label_val == 1)
                
                # if (is_af and not visualized_af):
                #     plt.figure(figsize=(12, 4))
                    
                #     # Thiết lập màu sắc và tiêu đề theo nhãn
                #     color = '#e74c3c' if is_af else '#27ae60'
                #     title = "REPRESENTATIVE: AFIB (Label 1)" if is_af else "REPRESENTATIVE: Non-AF/Normal (Label 0)"
                    
                #     time = np.arange(len(processed_segment)) / 32 # Tần số 32Hz của DeepBeat
                    
                #     plt.plot(time, processed_segment, color=color, linewidth=1.5)
                #     plt.title(f"{title} | Index: {i}", fontsize=12, fontweight='bold')
                #     plt.xlabel("Time (seconds)")
                #     plt.ylabel("Normalized PPG Amplitude")
                #     plt.grid(True, linestyle=':', alpha=0.7)
                #     plt.tight_layout()
                #     plt.show()

                #     # # Đánh dấu đã hiển thị xong để không lặp lại ở các segment sau
                #     # if is_af: visualized_af = True
                #     # else: visualized_non_af = True
              

            except Exception as e:
                print(f" Error processing at index {i} of {file_name}: {e}")
                continue
        global_stats['final_count'] = len(all_labels)
      

    if all_ppgs:
        final_ppgs = np.array(all_ppgs)
        final_labels = np.array(all_labels)
        
        # np.savez_compressed(SAVE_PATH, ppgs=final_ppgs, labels=final_labels)
        print("\n" + "="*60)
        print("BÁO CÁO CHI TIẾT LOẠI BỎ DỮ LIỆU (DETAILED DROPOUT REPORT)")
        print("="*60)
        print(f"1. Tổng segment nạp vào:            {global_stats['total_raw']}")
        
        print(f"\n2. Lọc lỗi vật lý (NaN/Flat Line):")
        print(f"   - Tổng số bị loại:               {global_stats['nan_flat_removed']}")
        print(f"   - Trong đó là nhãn AF:           {global_stats['nan_flat_af_removed']}")
        
        print(f"\n3. Lọc chất lượng tín hiệu (SQI < 0.3):")
        print(f"   - Tổng số bị loại:               {global_stats['quality_removed']}")
        print(f"   - Trong đó là nhãn AF:           {global_stats['quality_af_removed']}")

        print(f"Total samples obtained: {len(final_labels)}")
        print(f"AF rate (1): {np.sum(final_labels == 1) / len(final_labels):.2%}")
        print(f"Saved at: {SAVE_PATH}")
    else:
        print(" No data passed the filter.")

if __name__ == "__main__":
    combine_and_preprocess_datasets()

