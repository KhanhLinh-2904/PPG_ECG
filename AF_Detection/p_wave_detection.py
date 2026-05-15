import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
from scipy.fftpack import fft, fftfreq

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


def get_p_wave_features(signal, fs=125):
    
    r_indices = pan_tompkins_qrs(signal, fs)
    
    if len(r_indices) < 2:
        return 0.0, 0.0  
        
    search_start = int(0.45 * fs) 
    search_end = int(0.05 * fs)   
    
    nyq = 0.5 * fs
    b, a = butter(2, [0.5 / nyq, 15.0 / nyq], btype='band')
    clean_signal = filtfilt(b, a, signal)

    num_p_found = 0
    
    total_power_1_3 = 0.0
    total_power_3_10 = 0.0
    
    for r in r_indices:
        start_idx = r - search_start
        end_idx = r - search_end
        
        if start_idx >= 0 and end_idx < len(signal):
            p_window = clean_signal[start_idx:end_idx]
            p_detrended = p_window - np.mean(p_window) 
            
            peaks, _ = find_peaks(p_detrended, prominence=0.5)
            if len(peaks) > 0:
                num_p_found += 1
                
            windowed = p_detrended * np.hamming(len(p_detrended))

            N = len(windowed)
            freqs = fftfreq(N, 1/fs)[:N//2]
            energy = np.abs(fft(windowed))[:N//2] ** 2
            
            power_1_3Hz = np.sum(energy[(freqs >= 1.0) & (freqs < 3.0)])
            power_3_10Hz = np.sum(energy[(freqs >= 3.0)])
            
            total_power_1_3 += power_1_3Hz
            total_power_3_10 += power_3_10Hz

    p_ratio = num_p_found / len(r_indices)
    
    if total_power_1_3 == 0:
        total_power_1_3 = 1e-6
        
    avg_energy_ratio = total_power_3_10 / total_power_1_3
    
    return p_ratio, avg_energy_ratio