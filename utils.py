import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from biosppy.signals import ecg 
import neurokit2 as nk
from scipy.signal import resample

def detect_ecg_features(signal, sampling_rate=250):
    rpeaks_indices = ecg.hamilton_segmenter(signal=signal, sampling_rate=sampling_rate)['rpeaks']
   
    all_peaks, _ = find_peaks(signal, distance=sampling_rate*0.2) # Khoảng cách tối thiểu giữa các đỉnh ~200ms
    
    all_valleys, _ = find_peaks(-signal, distance=sampling_rate*0.2)

    opeaks_indices = np.setdiff1d(np.union1d(all_peaks, all_valleys), rpeaks_indices)
    
    return rpeaks_indices, opeaks_indices


def calculate_bsqi(ecg_signal, fs, tolerance_ms=150):
    _, info_hamilton = nk.ecg_peaks(ecg_signal, sampling_rate=fs, method="hamilton2002")
    peaks_hamilton = info_hamilton["ECG_R_Peaks"]
    _, info_zong = nk.ecg_peaks(ecg_signal, sampling_rate=fs, method="zong2003")
    peaks_zong = info_zong["ECG_R_Peaks"]
    peaks_alg1, peaks_alg2 = np.array(peaks_hamilton), np.array(peaks_zong)
    tolerance_samples = int((tolerance_ms / 1000.0) * fs)
    agreed_peaks = 0
    matched_in_alg2 = set()
    
    for p1 in peaks_alg1:
        matches = np.where((peaks_alg2 >= p1 - tolerance_samples) & 
                           (peaks_alg2 <= p1 + tolerance_samples))[0]
        for m in matches:
            if m not in matched_in_alg2:
                agreed_peaks += 1
                matched_in_alg2.add(m)
                break
                
    total_unique_peaks = len(peaks_alg1) + len(peaks_alg2) - agreed_peaks
    if total_unique_peaks == 0: return 0.0
    return agreed_peaks / total_unique_peaks

def calculate_sq_mask(ppg_signal, fs):

    N = len(ppg_signal)
    smoothing_window = 3 * fs
    template_length = 1 * fs
    signal_centered = ppg_signal - np.mean(ppg_signal)
  
    zero_crossings = np.where(np.diff(np.sign(signal_centered)))[0]
    
    if len(zero_crossings) < 2:
        return np.zeros(N)
        
    segments = []
    for i in range(len(zero_crossings) - 1):
        start_idx = zero_crossings[i]
        end_idx = zero_crossings[i+1]
        segments.append({
            'data': signal_centered[start_idx:end_idx],
            'start': start_idx,
            'end': end_idx
        })
        
    S_cc_interpolated = []
    S_cx_interpolated = []
    
    for seg in segments:
        seg_resampled = resample(seg['data'], template_length)
        seg['resampled'] = seg_resampled
        
        if np.mean(seg['data']) < 0:
            seg['type'] = 'cc'
            S_cc_interpolated.append(seg_resampled)
        else:
            seg['type'] = 'cx'
            S_cx_interpolated.append(seg_resampled)
            
    if len(S_cc_interpolated) > 0:
        T_cc = np.mean(np.array(S_cc_interpolated), axis=0)
    else:
        T_cc = np.zeros(template_length)
        
    if len(S_cx_interpolated) > 0:
        T_cx = np.mean(np.array(S_cx_interpolated), axis=0)
    else:
        T_cx = np.zeros(template_length)

    # Bước 5 & 6: Tính Pearson Correlation cho từng đoạn và gán vào Mask (M)
    M = np.zeros(N) # Khởi tạo mặt nạ toàn số 0 (Giải quyết luôn phần Padding M_0 và M_h)
    
    for seg in segments:
        S_prime = seg['resampled']
        T_prime = T_cc if seg['type'] == 'cc' else T_cx
        
        # Tránh lỗi chia cho 0 nếu template là đường thẳng
        if np.std(S_prime) == 0 or np.std(T_prime) == 0:
            corr = 0
        else:
            # Tính Pearson Correlation Coefficient
            corr = np.corrcoef(S_prime, T_prime)[0, 1]
            
        # Gán giá trị corr này cho toàn bộ chiều dài của đoạn (J_seg)
        M[seg['start']:seg['end']] = corr
        
    # Bước 7: Chuẩn hóa Min-Max (Min-Max Normalization) về [0, 1]
    m_min, m_max = np.min(M), np.max(M)
    if m_max > m_min:
        M_norm = (M - m_min) / (m_max - m_min)
    else:
        M_norm = np.zeros(N)
        
    # Bước 8: Làm mượt bằng Moving Average (300 points)
    window = np.ones(smoothing_window) / smoothing_window
    M_smoothed = np.convolve(M_norm, window, mode='same')
    
    return M_smoothed

def calculate_ppg_sqi(ppg_signal, fs, method='mean', threshold=0.5):
    N = len(ppg_signal)
    smoothing_window = 3 * fs
    template_length = 1 * fs
    signal_centered = ppg_signal - np.mean(ppg_signal)
    
    zero_crossings = np.where(np.diff(np.sign(signal_centered)))[0]
    
    # Nếu tín hiệu quá rác không tìm thấy đủ điểm cắt không, trả về SQI = 0.0
    if len(zero_crossings) < 2:
        return 0.0
        
    segments = []
    for i in range(len(zero_crossings) - 1):
        start_idx = zero_crossings[i]
        end_idx = zero_crossings[i+1]
        segments.append({
            'data': signal_centered[start_idx:end_idx],
            'start': start_idx,
            'end': end_idx
        })
        
    S_cc_interpolated = []
    S_cx_interpolated = []
    
    for seg in segments:
        seg_resampled = resample(seg['data'], template_length)
        seg['resampled'] = seg_resampled
        
        if np.mean(seg['data']) < 0:
            seg['type'] = 'cc'
            S_cc_interpolated.append(seg_resampled)
        else:
            seg['type'] = 'cx'
            S_cx_interpolated.append(seg_resampled)
            
    if len(S_cc_interpolated) > 0:
        T_cc = np.mean(np.array(S_cc_interpolated), axis=0)
    else:
        T_cc = np.zeros(template_length)
        
    if len(S_cx_interpolated) > 0:
        T_cx = np.mean(np.array(S_cx_interpolated), axis=0)
    else:
        T_cx = np.zeros(template_length)

    M = np.zeros(N) 
    
    for seg in segments:
        S_prime = seg['resampled']
        T_prime = T_cc if seg['type'] == 'cc' else T_cx
        
        if np.std(S_prime) == 0 or np.std(T_prime) == 0:
            corr = 0
        else:
            corr = np.corrcoef(S_prime, T_prime)[0, 1]
            
        M[seg['start']:seg['end']] = corr
        
    m_min, m_max = np.min(M), np.max(M)
    if m_max > m_min:
        M_norm = (M - m_min) / (m_max - m_min)
    else:
        M_norm = np.zeros(N)
        
    window = np.ones(smoothing_window) / smoothing_window
    M_smoothed = np.convolve(M_norm, window, mode='same')
   
    if method == 'mean':
        # Tính trung bình cộng của toàn bộ mặt nạ
        sqi_value = np.mean(M_smoothed)
        
    elif method == 'threshold':
        # Tính tỷ lệ phần trăm thời gian mà mặt nạ vượt qua ngưỡng an toàn
        sqi_value = np.sum(M_smoothed > threshold) / N
        
    else:
        sqi_value = np.mean(M_smoothed)
        
    return float(sqi_value)
if __name__ == "__main__":

    # fs = 250  
    # duration = 10  
    # clean_ecg = nk.ecg_simulate(duration=duration, sampling_rate=fs, heart_rate=70)

    
    # noise = nk.signal_distort(clean_ecg, sampling_rate=fs, 
    #                         noise_amplitude=0.2, 
    #                         artifacts_amplitude=0.1, 
    #                         artifacts_frequency=0.5)
    # noisy_ecg = clean_ecg + noise

    # _, info_hamilton = nk.ecg_peaks(noisy_ecg, sampling_rate=fs, method="pantompkins1985")
    # peaks_hamilton = info_hamilton["ECG_R_Peaks"]

    # _, info_zong = nk.ecg_peaks(noisy_ecg, sampling_rate=fs, method="zong2003")
    # peaks_zong = info_zong["ECG_R_Peaks"]

    # bsqi_score = calculate_bsqi(noisy_ecg, fs)
    # print("-" * 40)
    # print(f"Số đỉnh Hamilton bắt được: {len(peaks_hamilton)}")
    # print(f"Số đỉnh Zong (wqrs) bắt được: {len(peaks_zong)}")
    # print(f"Điểm chất lượng bSQI: {bsqi_score:.2f}")
    # print("-" * 40)

    # plt.figure(figsize=(12, 6))
    # plt.plot(noisy_ecg, label="Noisy ECG Signal", color='lightgrey')

    # plt.scatter(peaks_hamilton, noisy_ecg[peaks_hamilton], 
    #             color='blue', s=100, label='Hamilton (DF)', zorder=3)

    # plt.scatter(peaks_zong, noisy_ecg[peaks_zong], 
    #             color='red', marker='x', s=100, label='Zong 2003 (LT)', zorder=4)

    # plt.title(f"So sánh dò đỉnh R: Hamilton vs Zong 2003 (bSQI = {bsqi_score:.2f})")
    # plt.xlabel("Samples")
    # plt.ylabel("Amplitude")
    # plt.legend(loc="upper right")
    # plt.xlim(0, fs * 5) 
    # plt.tight_layout()
    # plt.show()

    fs = 100  # Tần số lấy mẫu: 100 Hz
    t = np.linspace(0, 30, 30 * fs)  # Tạo mảng thời gian 30 giây (3000 điểm)

    # Tạo tín hiệu PPG sạch giả lập (tổng hợp từ 2 sóng sin tạo hình dáng nhịp tim)
    clean_ppg = np.sin(2 * np.pi * 1.2 * t) + 0.4 * np.cos(2 * np.pi * 2.4 * t)

    # Cố tình tạo nhiễu cực mạnh từ giây 10 đến giây 15 (mô phỏng vung tay)
    noisy_ppg = np.copy(clean_ppg)
    noise_start, noise_end = 10 * fs, 15 * fs
    noisy_ppg[noise_start:noise_end] += np.random.normal(0, 1.5, noise_end - noise_start)

    # Gọi hàm tính SQ-Mask (Cửa sổ 300 điểm tương đương 3 giây làm mượt)
    sq_mask = calculate_sq_mask(noisy_ppg,fs)
    segment_sqi = np.mean(sq_mask)
    segment_sqi = np.sum(sq_mask > 0.5) / len(sq_mask)
    print("segment_sqi:", segment_sqi)
    # ==========================================
    # 3. TRỰC QUAN HÓA KẾT QUẢ
    # ==========================================
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Vẽ đường PPG (Trục Y bên trái)
    ax1.plot(t, noisy_ppg, color='steelblue', alpha=0.7, label='Tín hiệu PPG (Nhiễu giây 10-15)')
    ax1.set_xlabel('Thời gian (giây)')
    ax1.set_ylabel('Biên độ PPG', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')

    # Vẽ đường SQ-Mask (Trục Y bên phải)
    ax2 = ax1.twinx()
    ax2.plot(t, sq_mask, color='red', linewidth=2.5, label='Chỉ số SQ-Mask (0=Rác, 1=Sạch)')
    ax2.set_ylabel('Độ tin cậy của tín hiệu (SQ-Mask)', color='red', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(-0.1, 1.1)

    # Thêm ghi chú
    plt.axvspan(10, 15, color='gray', alpha=0.2, label='Vùng bị nhiễu động')
    fig.tight_layout()
    plt.title("Đánh giá tín hiệu PPG động bằng SQ-Mask (Template Matching)", fontsize=14)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()