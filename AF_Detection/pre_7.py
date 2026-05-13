import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
from sklearn.cluster import KMeans
import random
import torch

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =====================================================================
# 1. TIỀN XỬ LÝ & QRS DETECTOR (PAN-TOMPKINS)
# =====================================================================
def pan_tompkins_qrs(ecg_signal: np.ndarray, fs: int = 125):
    ecg_signal = np.array(ecg_signal).flatten()
    nyq = 0.5 * fs
    b, a = butter(1, [5.0 / nyq, 15.0 / nyq], btype='band')
    filtered_ecg = filtfilt(b, a, ecg_signal)
    
    diff_ecg = np.diff(filtered_ecg)
    diff_ecg = np.insert(diff_ecg, 0, diff_ecg[0])
    squared_ecg = diff_ecg ** 2
    
    window_width = int(0.15 * fs)
    integrated_ecg = np.convolve(squared_ecg, np.ones(window_width) / window_width, mode='same')
    
    threshold = np.mean(integrated_ecg)
    min_distance = int(0.3 * fs)
    peaks_integrated, _ = find_peaks(integrated_ecg, height=threshold, distance=min_distance)
    
    r_peaks = []
    search_window = int(0.05 * fs)
    for p in peaks_integrated:
        start = max(0, p - search_window)
        end = min(len(ecg_signal), p + search_window)
        if start < end:
            local_max = np.argmax(ecg_signal[start:end])
            r_peaks.append(start + local_max)
    return np.array(r_peaks)

# =====================================================================
# 2. HÀM TOÁN HỌC: MAXIMUM VERTICAL DISTANCE METHOD
# =====================================================================
def maximum_vertical_distance(signal_segment):
    """
    Tìm điểm có khoảng cách thẳng đứng lớn nhất tới đường nối 2 đầu mút.
    (Dùng để tìm QRS onset, P-onset, P-end theo bài báo)
    """
    if len(signal_segment) < 3:
        return 0
        
    x = np.arange(len(signal_segment))
    y = signal_segment
    
    # Tọa độ 2 đầu mút
    x1, y1 = x[0], y[0]
    x2, y2 = x[-1], y[-1]
    
    # Nếu đường nằm ngang hoặc 1 điểm
    if x2 == x1:
        return 0
        
    # Tính đường thẳng L(x) nối từ đầu đến cuối
    slope = (y2 - y1) / (x2 - x1)
    L_x = y1 + slope * (x - x1)
    
    # Tính khoảng cách thẳng đứng D(x) = |y(x) - L(x)|
    distances = np.abs(y - L_x)
    
    # Điểm có khoảng cách lớn nhất
    max_idx = np.argmax(distances)
    return max_idx

# =====================================================================
# 3. THUẬT TOÁN THEO BÀI BÁO (CLUSTERING & P-WAVE DETECTION)
# =====================================================================
def detect_p_wave_by_clustering(signal, fs=125):
    # 1. Split vào 30s segments (Bỏ qua vì demo_data thường đã là segment ngắn)
    
    # 2. QRS Peak Detection
    r_peaks = pan_tompkins_qrs(signal, fs=fs)
    if len(r_peaks) < 5:
        return None, None, None, None # Tín hiệu quá ngắn
        
    # 3. Trích xuất các nhịp (Beats aligned at QRS)
    win_left = int(0.3 * fs)  # Lùi 300ms
    win_right = int(0.2 * fs) # Tiến 200ms
    
    beats = []
    valid_r = []
    
    for r in r_peaks:
        start = r - win_left
        end = r + win_right
        if start >= 0 and end < len(signal):
            beats.append(signal[start:end])
            valid_r.append(r)
            
    beats = np.array(beats)
    
    if len(beats) < 5:
        return None, None, None, None
        
    # 4. Gom cụm bằng KMeans (Mô phỏng mạng 3x3 SOM = 9 neurons/clusters)
    n_clusters = min(9, len(beats)) # Đề phòng có ít hơn 9 nhịp
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(beats)
    
    # 5. Tìm Dominant Cluster (Cụm chứa nhiều nhịp nhất)
    unique_labels, counts = np.unique(labels, return_counts=True)
    dominant_label = unique_labels[np.argmax(counts)]
    
    dominant_beats = beats[labels == dominant_label]
    
    # Tính Correlation để gộp các cụm tương đồng (Mô phỏng logic gộp của SOM)
    avg_dominant = np.mean(dominant_beats, axis=0)
    merged_beats = list(dominant_beats)
    
    for lbl in unique_labels:
        if lbl != dominant_label:
            cluster_avg = np.mean(beats[labels == lbl], axis=0)
            corr = np.corrcoef(avg_dominant, cluster_avg)[0, 1]
            if corr > 0.90:  # Ngưỡng High Correlation
                merged_beats.extend(beats[labels == lbl])
                
    merged_beats = np.array(merged_beats)
    
    # 6. Tính Dominant Average Beat (Nhịp trung bình không chứa nhiễu)
    avg_beat = np.mean(merged_beats, axis=0)
    
    # 7. Smoothing bằng Moving Average Filter
    ma_window = int(0.02 * fs) # 20ms
    avg_beat_smooth = np.convolve(avg_beat, np.ones(ma_window)/ma_window, mode='same')
    
    # ================= ĐO ĐẠC TRÊN AVERAGE BEAT =================
    r_idx_local = win_left # Đỉnh R luôn nằm ở vị trí win_left do cách ta cắt
    
    # Tìm Qo (QRS onset) bằng Maximum Vertical Distance
    qo_search_start = r_idx_local - int(0.1 * fs) # Tìm trong 100ms trước R
    qo_local_offset = maximum_vertical_distance(avg_beat_smooth[qo_search_start:r_idx_local])
    qo_idx = qo_search_start + qo_local_offset
    
    # Cửa sổ tìm P-wave
    p_search_start = 0 
    p_search_end = qo_idx - int(0.02 * fs) # Chừa khoảng an toàn 20ms
    
    if p_search_end > p_search_start:
        p_window_raw = avg_beat_smooth[p_search_start:p_search_end]
        
        # --- FIX: KHỬ XU HƯỚNG ĐƯỜNG NỀN (DETRENDING) ---
        # Bẻ phẳng cửa sổ bằng cách trừ đi đường thẳng nối 2 đầu mút
        x_vals = np.arange(len(p_window_raw))
        y1, y2 = p_window_raw[0], p_window_raw[-1]
        slope = (y2 - y1) / len(p_window_raw) if len(p_window_raw) > 1 else 0
        baseline_trend = y1 + slope * x_vals
        p_window_flat = p_window_raw - baseline_trend
        
        # Tìm Pp trên tín hiệu đã được BẺ PHẲNG
        pp_local_offset = np.argmax(np.abs(p_window_flat)) 
        pp_idx = p_search_start + pp_local_offset
        
        # Tìm Po và Pe bằng MVD trên TÍN HIỆU GỐC (để ranh giới bám sát đường cong thật)
        po_local_offset = maximum_vertical_distance(avg_beat_smooth[p_search_start:pp_idx+1])
        po_idx = p_search_start + po_local_offset
        
        pe_local_offset = maximum_vertical_distance(avg_beat_smooth[pp_idx:qo_idx])
        pe_idx = pp_idx + pe_local_offset
    else:
        pp_idx, po_idx, pe_idx = 0, 0, 0 # Không tìm thấy
        
    # ================= DÒ TÌM TRÊN TỪNG NHỊP CỤ THỂ =================
    # Sử dụng "Smaller peak search window" quanh Pp trung bình
    individual_p_peaks = []
    
    p_window_narrow_start = max(0, pp_idx - int(0.04 * fs))
    p_window_narrow_end = min(win_left, pp_idx + int(0.04 * fs))
    
    for r in valid_r:
        if pp_idx != 0:
            # Tọa độ tuyệt đối trên tín hiệu gốc
            abs_search_start = (r - win_left) + p_window_narrow_start
            abs_search_end = (r - win_left) + p_window_narrow_end
            
            if abs_search_start >= 0 and abs_search_end < len(signal):
                small_window = signal[abs_search_start:abs_search_end]
                local_p_peak = np.argmax(np.abs(small_window - np.mean(small_window)))
                abs_p_peak = abs_search_start + local_p_peak
                individual_p_peaks.append(abs_p_peak)
                
    return avg_beat_smooth, (po_idx, pp_idx, pe_idx), valid_r, individual_p_peaks

# =====================================================================
# 4. HÀM HIỂN THỊ KIỂM CHỨNG
# =====================================================================
def visualize_paper_method(dataset_path):
    if not os.path.exists(dataset_path):
        print(f"[!] Lỗi: Không tìm thấy file {dataset_path}")
        return

    data = np.load(dataset_path, allow_pickle=True)
    ecgs = data["ecgs"]
    labels = data["labels"]
    
    print("\n" + "="*80)
    print(" HỆ THỐNG MÔ PHỎNG P-WAVE DETECTION (DỰA TRÊN SOM & MVD)")
    print("="*80)
    
    for i in range(min(2000000, len(ecgs))):
        sig = ecgs[i]
        lbl = labels[i]
        lbl_str = "AF" if lbl == 1 else "Non-AF"
        
        # Chỉ hiển thị nhịp bình thường để dễ kiểm chứng sóng P
        if lbl == 1: 
            print("ok")
            avg_beat, p_landmarks, r_peaks, ind_p_peaks = detect_p_wave_by_clustering(sig, fs=125)
            print(avg_beat , p_landmarks[1])
            if avg_beat is not None and p_landmarks[1] != 0:
                po_idx, pp_idx, pe_idx = p_landmarks
                
                # ÉP KIỂU SANG NUMPY ARRAY ĐỂ SỬA LỖI TOÁN HỌC VECTOR
                r_peaks = np.array(r_peaks) 
                
                fig, axs = plt.subplots(1, 2, figsize=(14, 5))
                
                # Biểu đồ 1: DOMINANT AVERAGE BEAT
                t_avg = np.arange(len(avg_beat)) / 125 - 0.3 # Trục X: 0 là đỉnh R
                axs[0].plot(t_avg, avg_beat, color='black', linewidth=1.5)
                axs[0].scatter(t_avg[pp_idx], avg_beat[pp_idx], color='limegreen', s=80, label='Pp (P-peak)', zorder=4)
                axs[0].scatter(t_avg[po_idx], avg_beat[po_idx], color='blue', marker='>', s=80, label='Po (P-onset)', zorder=4)
                axs[0].scatter(t_avg[pe_idx], avg_beat[pe_idx], color='magenta', marker='<', s=80, label='Pe (P-end)', zorder=4)
                axs[0].axvspan(t_avg[po_idx], t_avg[pe_idx], color='gold', alpha=0.3)
                
                axs[0].set_title("Dominant Average Beat & MVD Delineation", fontweight='bold')
                axs[0].set_xlabel("Thời gian so với đỉnh R (giây)")
                axs[0].set_ylabel("Amplitude")
                axs[0].legend()
                axs[0].grid(True, alpha=0.3)
                
                # Biểu đồ 2: ÁP DỤNG CỬA SỔ NHỎ LÊN TÍN HIỆU GỐC
                t_sig = np.arange(len(sig)) / 125
                axs[1].plot(t_sig, sig, color='gray', linewidth=1, alpha=0.8)
                
                # Hàm scatter giờ đã nhận np.array, sẽ không còn lỗi chia cho int
                axs[1].scatter(r_peaks/125, sig[r_peaks], color='red', marker='v', label='R-peaks', zorder=3)
                
                if len(ind_p_peaks) > 0:
                    ind_p_peaks = np.array(ind_p_peaks)
                    axs[1].scatter(ind_p_peaks/125, sig[ind_p_peaks], color='limegreen', marker='o', s=40, label='Ind. P-peaks', zorder=4)
                    
                axs[1].set_title(f"Record {i} [{lbl_str}] - Individual P-peaks (Smaller Search Window)", fontweight='bold')
                axs[1].set_xlabel("Thời gian (giây)")
                axs[1].legend(loc='upper right')
                axs[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.show(block=True)

if __name__ == "__main__":
    set_seed(42)
    dataset_path = "/home/linhhima/PPG_ECG/datasets/z_score_norm/MIT_BIH_test_segments.npz"
    visualize_paper_method(dataset_path)