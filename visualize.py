import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List

# --- CẤU HÌNH HIỂN THỊ (Bạn có thể chỉnh sửa tại đây) ---
FS = 125             # Tần số lấy mẫu giả định (Hz)
WINDOW_SECONDS = 10  # Độ rộng cửa sổ hiển thị (giây)
WINDOW_SAMPLES = int(FS * WINDOW_SECONDS)
STEP_SIZE = 10       # Tốc độ trượt (số mẫu dịch chuyển mỗi khung hình). Tăng lên = nhanh hơn.
PAUSE_TIME = 0.001   # Thời gian nghỉ giữa các khung (giây)

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
    """
    total_samples = len(ppg_signal)
    
    # Nếu bản ghi ngắn hơn cửa sổ hiển thị, không chạy được
    if total_samples < WINDOW_SAMPLES:
        print(f"⚠️ Bản ghi {record_name} quá ngắn ({total_samples} mẫu) so với cửa sổ ({WINDOW_SAMPLES} mẫu). Bỏ qua.")
        return

    # 1. Thiết lập Figure và Axes
    plt.ion() # Bật chế độ tương tác
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f"Streaming Record: {record_name} (FS={fs}Hz)", fontsize=16)

    # Tạo trục x (thời gian tương đối 0 -> WINDOW_SECONDS)
    x_axis = np.linspace(0, WINDOW_SECONDS, WINDOW_SAMPLES)

    # Khởi tạo các line plot với dữ liệu rỗng ban đầu
    # Chúng ta sẽ cập nhật dữ liệu của các line này trong vòng lặp
    line_ppg, = ax1.plot(x_axis, np.zeros(WINDOW_SAMPLES), color='blue', linewidth=1.5)
    line_ecg, = ax2.plot(x_axis, np.zeros(WINDOW_SAMPLES), color='red', linewidth=1.5)

    # --- Thiết lập giới hạn trục Y cố định ---
    # Điều này cực kỳ quan trọng để biểu đồ không bị "nhảy" (jitter) khi biên độ thay đổi
    # Ta tính min/max trên toàn bộ tín hiệu để cố định khung hình
    margin_ppg = (np.max(ppg_signal) - np.min(ppg_signal)) * 0.1
    margin_ecg = (np.max(ecg_signal) - np.min(ecg_signal)) * 0.1
    
    # Fallback nếu tín hiệu là đường thẳng (max = min)
    if margin_ppg == 0: margin_ppg = 1.0
    if margin_ecg == 0: margin_ecg = 1.0

    ax1.set_ylim(np.min(ppg_signal) - margin_ppg, np.max(ppg_signal) + margin_ppg)
    ax2.set_ylim(np.min(ecg_signal) - margin_ecg, np.max(ecg_signal) + margin_ecg)

    # Trang trí
    ax1.set_ylabel("PPG Amplitude")
    ax1.set_title("PPG Signal (Blue)")
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.set_ylabel("ECG Amplitude")
    ax2.set_xlabel("Time in Window (seconds)")
    ax2.set_title("ECG Signal (Red)")
    ax2.grid(True, linestyle='--', alpha=0.6)

    print(f"▶️ Đang phát bản ghi: {record_name}...")
    print("   (Đóng cửa sổ đồ thị để chuyển sang record tiếp theo)")

    # 2. Vòng lặp trượt dữ liệu (Sliding Loop)
    # Duyệt từ đầu đến cuối file với bước nhảy STEP_SIZE
    try:
        # range(start, stop, step)
        for start_idx in range(0, total_samples - WINDOW_SAMPLES, STEP_SIZE):
            
            # Kiểm tra nếu người dùng đã đóng cửa sổ
            if not plt.fignum_exists(fig.number):
                print("⏹️ Cửa sổ đã bị đóng. Dừng phát record này.")
                return

            end_idx = start_idx + WINDOW_SAMPLES
            
            # Cắt lát dữ liệu hiện tại (Sliding Window)
            current_ppg = ppg_signal[start_idx:end_idx]
            current_ecg = ecg_signal[start_idx:end_idx]

            # Cập nhật dữ liệu cho đường vẽ (Line)
            line_ppg.set_ydata(current_ppg)
            line_ecg.set_ydata(current_ecg)

            # Vẽ lại canvas
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            
            # Tạm dừng một chút để mắt người kịp nhìn (Animation speed control)
            if PAUSE_TIME > 0:
                time.sleep(PAUSE_TIME)
            
    except KeyboardInterrupt:
        print("\n⏹️ Đã dừng phát record này do người dùng nhấn Ctrl+C.")
        plt.close(fig)
        return
    
    # Đóng figure sau khi chạy hết record này để chuẩn bị cho record sau
    plt.close(fig)
    print(f"✅ Đã hiển thị xong bản ghi {record_name}.")


def main_loop(datapath: str, record_list: List[str]):
    """
    Duyệt qua danh sách record và hiển thị từng cái.
    """
    print("="*60)
    print(f"CHẾ ĐỘ VISUALIZE: TRƯỢT TÍN HIỆU (SLIDING WINDOW)")
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
            # (Lưu ý: Cần kiểm tra dataset thực tế, đôi khi ngược lại)
            ppg = signal_data[:, 0]
            ecg = signal_data[:, 1]
            # ppg = normalize_signal(ppg)
            # ecg = normalize_signal(ecg)
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