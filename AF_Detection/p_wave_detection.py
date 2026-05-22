import numpy as np
from scipy.signal import find_peaks, butter, filtfilt

fs = 125
def pan_tompkins_qrs(ecg_signal: np.ndarray, external_threshold: float = None, return_threshold: bool = False):
    """Thuật toán Pan-Tompkins thay thế cho wfdb.xqrs_detect"""
    ecg_signal = np.array(ecg_signal).flatten()
    nyq = 0.5 * fs
    b, a = butter(1, [5.0 / nyq, 15.0 / nyq], btype='band')
    filtered_ecg = filtfilt(b, a, ecg_signal)
    
    diff_ecg = np.diff(filtered_ecg)
    diff_ecg = np.insert(diff_ecg, 0, diff_ecg[0])
    squared_ecg = diff_ecg ** 2
    
    window_width = int(0.15 * fs)
    integrated_ecg = np.convolve(squared_ecg, np.ones(window_width) / window_width, mode='same')
    
    # SỬ DỤNG NGƯỠNG TỪ BÊN NGOÀI NẾU CÓ
    if external_threshold is not None:
        threshold = external_threshold
    else:
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
            
    # TRẢ VỀ CẢ ĐỈNH R VÀ NGƯỠNG ĐÃ DÙNG NẾU CẦN
    if return_threshold:
        return np.array(r_peaks), threshold
        
    return np.array(r_peaks)

def get_p_ratio(signal, fs=125, gt_threshold=None):
   
    r_indices = pan_tompkins_qrs(signal, external_threshold=gt_threshold)
    
    if len(r_indices) < 2:
        return 0.0  
        
    nyq = 0.5 * fs
    b, a = butter(2, [0.5 / nyq, 15.0 / nyq], btype='band')
    clean_signal = filtfilt(b, a, signal)

    num_p_found = 0
    
    for i in range(1, len(r_indices)):
        r = r_indices[i]
        r_prev = r_indices[i-1] 
        
        rr_distance = r - r_prev
        
        search_start = int(rr_distance * 0.5)
        search_end = int(0.05 * rr_distance)
        
        start_idx = r - search_start
        end_idx = r - search_end
        
        if start_idx >= r_prev and end_idx < len(signal):
            p_window = clean_signal[start_idx:end_idx]
            p_detrended = p_window - np.mean(p_window)
            
            peaks, _ = find_peaks(p_detrended, prominence=0.5)
            if len(peaks) > 0:
                num_p_found += 1

    valid_beats = len(r_indices) - 1
    p_ratio = num_p_found / valid_beats if valid_beats > 0 else 0.0
    
    return p_ratio