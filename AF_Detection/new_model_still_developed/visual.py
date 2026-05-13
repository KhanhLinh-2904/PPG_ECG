import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
DATA_PATH = '/home/linhhima/PPG_ECG/AF_Detection/new_model_still_developed/reconstructed_data.npz'

def visualize_reconstructed_data(file_path, num_samples_to_show=10):
    """
    Đọc file .npz và vẽ biểu đồ tín hiệu PPG cùng với ECG được tái tạo.
    """
    if not os.path.exists(file_path):
        print(f"[!] Lỗi: Không tìm thấy file tại {file_path}")
        return

    print(f"[*] Đang tải dữ liệu từ {file_path}...")
    try:
        data = np.load(file_path, allow_pickle=True)
        
        ecgs = data['ecgs']      # Predicted ECGs
        ppgs = data['ppgs']      # Input PPGs
        labels = data['labels']  # Labels
        records = data['records'] # Record names
        
        total_samples = len(ecgs)
        print(f"[+] Đã tải thành công. Tổng số mẫu: {total_samples}")
        
    except Exception as e:
        print(f"[!] Lỗi khi đọc file: {e}")
        return

    # Xác định số lượng mẫu sẽ hiển thị
    samples_to_show = min(num_samples_to_show, total_samples)
    print(f"[*] Sẽ hiển thị {samples_to_show} mẫu đầu tiên. (Đóng cửa sổ ảnh để xem mẫu tiếp theo)")

    for i in range(samples_to_show):
        ppg_signal = ppgs[i].squeeze()
        ecg_signal = ecgs[i].squeeze()
        label = labels[i]
        
        # Xử lý trường hợp mảng records rỗng
        if records.size > 0 and i < len(records):
            record_name = records[i]
        else:
            record_name = f"Unknown_Record_{i}"

        # Trục thời gian
        t = np.arange(len(ecg_signal))

        # Khởi tạo biểu đồ
        plt.figure(figsize=(12, 6))
        plt.suptitle(f"Record: {record_name} | Label: {label}", fontsize=14, fontweight='bold')

        # Biểu đồ 1: Input PPG
        plt.subplot(2, 1, 1)
        plt.plot(t, ppg_signal, color='green', label='Input PPG')
        plt.title("Input PPG Signal")
        plt.ylabel("Amplitude")
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)

        # Biểu đồ 2: Reconstructed ECG
        plt.subplot(2, 1, 2)
        plt.plot(t, ecg_signal, color='red', label='Reconstructed ECG')
        plt.title("Predicted/Reconstructed ECG")
        plt.xlabel("Time Samples")
        plt.ylabel("Amplitude")
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        
        # Hiển thị biểu đồ (Code sẽ tạm dừng ở đây cho đến khi bạn tắt cửa sổ ảnh)
        plt.show()

if __name__ == "__main__":
    # Thay đổi num_samples_to_show thành số lượng mẫu bạn muốn xem
    visualize_reconstructed_data(DATA_PATH, num_samples_to_show=100)