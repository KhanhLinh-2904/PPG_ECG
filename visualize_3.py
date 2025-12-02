import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List
from scipy import signal

# --- CẤU HÌNH HIỂN THỊ ---
FS = 125             
WINDOW_SECONDS = 10  
WINDOW_SAMPLES = int(FS * WINDOW_SECONDS)
STEP_SIZE = 10       
PAUSE_TIME = 0.001   

# --- CẤU HÌNH XỬ LÝ TÍN HIỆU ---
INVERT_ECG = False  # <--- Đặt thành True nếu đỉnh ECG bị ngược

def normalize_signal(signal):
    """Chuẩn hóa min-max tín hiệu về [0, 1]."""
    min_val = np.min(signal)
    max_val = np.max(signal)
    if max_val - min_val < 1e-8:
        return np.zeros_like(signal)
    return (signal - min_val) / (max_val - min_val)

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

def get_all_records(datapath: str) -> List[str]:
    if not os.path.exists(datapath):
        print(f"❌ Lỗi: Đường dẫn dữ liệu không tồn tại: {datapath}")
        return []
    records = [f.replace('.hea', '') for f in os.listdir(datapath) if f.endswith('.hea')]
    if not records:
         records = [f.split('.')[0] for f in os.listdir(datapath) if f.endswith('.dat')]
    return sorted(list(set(records)))

def visualize_sliding_record(record_name: str, ppg_signal: np.ndarray, ecg_signal: np.ndarray, fs: int):
    """
    Hiển thị hiệu ứng trượt với tính năng CĂN CHỈNH và ĐẢO NGƯỢC ECG.
    """
    total_samples = len(ppg_signal)
    if total_samples < WINDOW_SAMPLES:
        print(f"⚠️ Bản ghi {record_name} quá ngắn. Bỏ qua.")
        return

    # --- 1. ĐẢO NGƯỢC ECG (NẾU CẦN) ---
    if INVERT_ECG:
        print("🔄 Đang đảo ngược tín hiệu ECG (Inverting)...")
        ecg_signal = -ecg_signal 
        # Sau khi đảo ngược, cần chuẩn hóa lại để về [0, 1]
        ecg_signal = normalize_signal(ecg_signal)

    # # --- 2. CĂN CHỈNH TÍN HIỆU ---
    # print("⏳ Đang tính toán căn chỉnh tín hiệu (Cross-Correlation)...")
    # ppg_signal_aligned, lag = align_signals_cross_correlation(ecg_signal, ppg_signal)

    plt.ion() 
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    title_text = f"Record: {record_name} (Aligned | Lag: )"
    if INVERT_ECG:
        title_text += " | ECG Inverted"
    fig.suptitle(title_text, fontsize=16)

    x_axis = np.linspace(0, WINDOW_SECONDS, WINDOW_SAMPLES)

    # Vẽ ECG và PPG
    line_ppg, = ax.plot(x_axis, np.zeros(WINDOW_SAMPLES), color='blue', linewidth=1.5, label=f'PPG (Shifted)', alpha=0.8)
    line_ecg, = ax.plot(x_axis, np.zeros(WINDOW_SAMPLES), color='red', linewidth=1.5, label='ECG', alpha=0.8)

    ax.set_ylim(-0.1, 1.1)
    ax.set_ylabel("Normalized Amplitude (0-1)")
    ax.set_xlabel("Time in Window (seconds)")
    ax.legend(loc="upper right") 
    ax.grid(True, linestyle='--', alpha=0.6)

    print(f"▶️ Đang phát bản ghi: {record_name}...")

    try:
        for start_idx in range(0, total_samples - WINDOW_SAMPLES, STEP_SIZE):
            if not plt.fignum_exists(fig.number):
                print("⏹️ Cửa sổ đã bị đóng.")
                return

            end_idx = start_idx + WINDOW_SAMPLES
            
            current_ppg = ppg_signal[start_idx:end_idx]
            current_ecg = ecg_signal[start_idx:end_idx] # Đã đảo ngược và chuẩn hóa

            line_ppg.set_ydata(current_ppg)
            line_ecg.set_ydata(current_ecg)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            
            if PAUSE_TIME > 0:
                time.sleep(PAUSE_TIME)
            
    except KeyboardInterrupt:
        print("\n⏹️ Đã dừng phát.")
        plt.close(fig)
        return
    
    plt.close(fig)
    print(f"✅ Hoàn tất bản ghi {record_name}.")

def main_loop(datapath: str, record_list: List[str]):
    print("="*60)
    print(f"CHẾ ĐỘ VISUALIZE: TRƯỢT TÍN HIỆU (AUTO-ALIGN + INVERT)")
    print("="*60)
    
    for i, record_name in enumerate(record_list):
        print(f"\n>>> [{i+1}/{len(record_list)}] Đang tải: {record_name}")
        record_path = os.path.join(datapath, record_name)
        try:
            record = wfdb.rdrecord(record_path)
            signal_data = record.p_signal
            if signal_data is None or signal_data.shape[1] < 2: continue
            
            ppg = signal_data[:, 0]
            ecg = signal_data[:, 1]
            
            # Chuẩn hóa ban đầu (trước khi xử lý)
            ppg = normalize_signal(ppg)
            ecg = normalize_signal(ecg)
            
            visualize_sliding_record(record_name, ppg, ecg, record.fs)
            time.sleep(1.0)

        except Exception as e:
            print(f"❌ Lỗi: {e}")
            continue

if __name__ == '__main__':
    YOUR_DATA_PATH = "/home/linhhima/Pre_processing_data/Datasets/mimic_perform_non_af_wfdb" 
    all_records = get_all_records(YOUR_DATA_PATH)
    records_to_show = all_records[:16]
    
    if not records_to_show:
        print(f"Không tìm thấy dữ liệu trong {YOUR_DATA_PATH}")
    else:
        main_loop(YOUR_DATA_PATH, records_to_show)