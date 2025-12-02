import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List
from scipy import signal
# --- CẤU HÌNH HIỂN THỊ (Bạn có thể chỉnh sửa tại đây) ---
FS = 125             # Tần số lấy mẫu giả định (Hz)
WINDOW_SECONDS = 10  # Độ rộng cửa sổ hiển thị (giây)
WINDOW_SAMPLES = int(FS * WINDOW_SECONDS)
STEP_SIZE = 10       # Tốc độ trượt (số mẫu dịch chuyển mỗi khung hình). Tăng lên = nhanh hơn.
PAUSE_TIME = 0.001   # Thời gian nghỉ giữa các khung (giây)
def align_signals_cross_correlation(ecg, ppg):
    """
    Căn chỉnh PPG theo ECG sử dụng Cross-Correlation.
    """
    # Trừ mean để loại bỏ DC offset
    correlation = signal.correlate(ecg - np.mean(ecg), ppg - np.mean(ppg), mode="full")
    lags = signal.correlation_lags(len(ecg), len(ppg), mode="full")
    optimal_lag = lags[np.argmax(correlation)]
    
    print(f"   -> Đã tìm thấy độ trễ (Lag): {optimal_lag} mẫu")
    
    # Dịch chuyển PPG
    aligned_ppg = np.roll(ppg, shift=optimal_lag)
    return aligned_ppg, optimal_lag
def normalize_signal(signal):
    """Chuẩn hóa min-max tín hiệu về [0, 1]."""
    min_val = np.min(signal)
    max_val = np.max(signal)
    if max_val - min_val < 1e-8:
        return np.zeros_like(signal)
    return (signal - min_val) / (max_val - min_val)

def get_all_records(datapath: str) -> List[str]:
    """
    Lấy danh sách tên các bản ghi (bỏ đuôi mở rộng) từ thư mục.
    Ưu tiên tìm file .hea, nếu không có thì tìm .dat.
    """
    if not os.path.exists(datapath):
        print(f"❌ Lỗi: Đường dẫn dữ liệu không tồn tại: {datapath}")
        return []
    
    # Lấy các file .hea (header) để xác định record
    records = [f.replace('.hea', '') for f in os.listdir(datapath) if f.endswith('.hea')]
    
    # Nếu không tìm thấy .hea, thử tìm .dat và bỏ đuôi
    if not records:
         records = [f.split('.')[0] for f in os.listdir(datapath) if f.endswith('.dat')]
    
    # Loại bỏ trùng lặp và sắp xếp
    return sorted(list(set(records)))

def visualize_sliding_record(record_name: str, ppg_signal: np.ndarray, ecg_signal: np.ndarray, fs: int):
    """
    Hiển thị hiệu ứng trượt (sliding/streaming) cho một bản ghi đơn lẻ.
    Dữ liệu sẽ trôi từ phải sang trái trên cửa sổ cố định.
    Cả PPG và ECG đều được vẽ trên CÙNG MỘT TRỤC TỌA ĐỘ.
    """
    total_samples = len(ppg_signal)
    
    # Nếu bản ghi ngắn hơn cửa sổ hiển thị, không chạy được
    if total_samples < WINDOW_SAMPLES:
        print(f"⚠️ Bản ghi {record_name} quá ngắn ({total_samples} mẫu) so với cửa sổ ({WINDOW_SAMPLES} mẫu). Bỏ qua.")
        return

    # 1. Thiết lập Figure và Axes
    plt.ion() # Bật chế độ tương tác
    
    # --- THAY ĐỔI: Chỉ tạo 1 subplot thay vì 2 ---
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    fig.suptitle(f"Streaming Record: {record_name} (FS={fs}Hz) - Normalized", fontsize=16)

    # Tạo trục x (thời gian tương đối 0 -> WINDOW_SECONDS)
    x_axis = np.linspace(0, WINDOW_SECONDS, WINDOW_SAMPLES)

    # Khởi tạo các line plot trên CÙNG MỘT TRỤC (ax)
    # Thêm alpha (độ trong suốt) để dễ nhìn khi chồng lên nhau
    line_ppg, = ax.plot(x_axis, np.zeros(WINDOW_SAMPLES), color='blue', linewidth=1.5, label='PPG', alpha=0.8)
    line_ecg, = ax.plot(x_axis, np.zeros(WINDOW_SAMPLES), color='red', linewidth=1.5, label='ECG', alpha=0.8)

    # --- Thiết lập giới hạn trục Y ---
    # Vì dữ liệu đã được normalize về [0, 1], ta có thể cố định trục Y
    ax.set_ylim(-0.1, 1.1)

    # Trang trí
    ax.set_ylabel("Normalized Amplitude (0-1)")
    ax.set_xlabel("Time in Window (seconds)")
    ax.legend(loc="upper right") # Hiển thị chú thích
    ax.grid(True, linestyle='--', alpha=0.6)

    print(f"▶️ Đang phát bản ghi: {record_name}...")
    print("   (Đóng cửa sổ đồ thị để chuyển sang record tiếp theo)")

    # 2. Vòng lặp trượt dữ liệu (Sliding Loop)
    try:
        for start_idx in range(0, total_samples - WINDOW_SAMPLES, STEP_SIZE):
            
            # Kiểm tra nếu người dùng đã đóng cửa sổ
            if not plt.fignum_exists(fig.number):
                print("⏹️ Cửa sổ đã bị đóng. Dừng phát record này.")
                return

            end_idx = start_idx + WINDOW_SAMPLES
            
            # Cắt lát dữ liệu hiện tại
            current_ppg = ppg_signal[start_idx:end_idx]
            current_ecg = ecg_signal[start_idx:end_idx]

            # Cập nhật dữ liệu cho đường vẽ
            line_ppg.set_ydata(current_ppg)
            line_ecg.set_ydata(current_ecg)

            # Vẽ lại canvas
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            
            if PAUSE_TIME > 0:
                time.sleep(PAUSE_TIME)
            
    except KeyboardInterrupt:
        print("\n⏹️ Đã dừng phát record này do người dùng nhấn Ctrl+C.")
        plt.close(fig)
        return
    
    # Đóng figure sau khi chạy hết record này
    plt.close(fig)
    print(f"✅ Đã hiển thị xong bản ghi {record_name}.")


def main_loop(datapath: str, record_list: List[str]):
    """
    Duyệt qua danh sách record và hiển thị từng cái.
    """
    print("="*60)
    print(f"CHẾ ĐỘ VISUALIZE: TRƯỢT TÍN HIỆU (CHỒNG LỚP)")
    print(f"Thư mục dữ liệu: {datapath}")
    print(f"Số lượng bản ghi: {len(record_list)}")
    print(f"Window: {WINDOW_SECONDS}s | Speed: {STEP_SIZE} samples/frame")
    print("="*60)
    
    for i, record_name in enumerate(record_list):
        print(f"\n>>> [{i+1}/{len(record_list)}] Đang tải và chuẩn bị: {record_name}")
        
        record_path = os.path.join(datapath, record_name)
        
        try:
            # Đọc bản ghi bằng WFDB
            record = wfdb.rdrecord(record_path)
            
            # Lấy tín hiệu
            signal_data = record.p_signal
            
            # Kiểm tra số kênh
            if signal_data is None or signal_data.shape[1] < 2:
                print(f"❌ Lỗi: Bản ghi {record_name} không có đủ dữ liệu hoặc số kênh < 2.")
                continue
            
            # Giả định kênh: Cột 0 là PPG, Cột 1 là ECG
            ppg = signal_data[:, 0]
            ecg = signal_data[:, 1]
            
            # --- CHUẨN HÓA (QUAN TRỌNG KHI VẼ CHUNG 1 TRỤC) ---
            ppg = normalize_signal(ppg)
            ecg = normalize_signal(ecg)
            
            # Gọi hàm hiển thị
            visualize_sliding_record(record_name, ppg, ecg, record.fs)
            
            # Nghỉ một chút giữa các record
            time.sleep(1.0)

        except Exception as e:
            print(f"❌ Lỗi ngoại lệ với bản ghi {record_name}: {e}")
            continue

if __name__ == '__main__':
    
    # --- !!! CẤU HÌNH ĐƯỜNG DẪN CỦA BẠN !!! ---
    YOUR_DATA_PATH = "/home/linhhima/Pre_processing_data/Datasets/mimic_perform_non_af_wfdb" 
    
    # Lấy danh sách tất cả record
    all_records = get_all_records(YOUR_DATA_PATH)
    
    # Chỉ lấy 16 record đầu tiên để demo
    records_to_show = all_records[:16]
    
    if not records_to_show:
        print(f"Không tìm thấy file dữ liệu nào trong {YOUR_DATA_PATH}")
    else:
        main_loop(YOUR_DATA_PATH, records_to_show)