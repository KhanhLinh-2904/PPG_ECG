import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from biosppy.signals import ecg 
import neurokit2 as nk
from scipy.signal import resample

def detect_ecg_features(signal, sampling_rate=250):
    rpeaks_indices = ecg.hamilton_segmenter(signal=signal, sampling_rate=sampling_rate)['rpeaks']
   
    all_peaks, _ = find_peaks(signal, distance=sampling_rate*0.2) 
    
    all_valleys, _ = find_peaks(-signal, distance=sampling_rate*0.2)

    opeaks_indices = np.setdiff1d(np.union1d(all_peaks, all_valleys), rpeaks_indices)
    
    return rpeaks_indices, opeaks_indices


def calculate_bsqi(ecg_signal, fs, tolerance_ms=150):
    pad_val = 500
    ecg_signal = np.pad(ecg_signal, (pad_val, pad_val), mode='reflect')
    try:
        _, info_hamilton = nk.ecg_peaks(ecg_signal, sampling_rate=fs, method="hamilton2002")
        peaks_hamilton = info_hamilton["ECG_R_Peaks"]
        _, info_zong = nk.ecg_peaks(ecg_signal, sampling_rate=fs, method="zong2003")
        peaks_zong = info_zong["ECG_R_Peaks"]
        # print("Peaks detected by Hamilton: ", len(peaks_hamilton))
        # print("Peaks detected by Zong: ", len(peaks_zong))
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
        if total_unique_peaks == 0:
            return 0.0
        return agreed_peaks / total_unique_peaks
    except Exception as e:
        print(f"Error in NeuroKit2 peak detection: {e}", flush=True)
        return 0.0

def visualize_bsqi_steps(ecg_signal, fs, tolerance_ms=150, record_name="ECG Sample"):
    _, info_hamilton = nk.ecg_peaks(ecg_signal, sampling_rate=fs, method="hamilton2002")
    peaks_hamilton = np.array(info_hamilton["ECG_R_Peaks"])
    
    _, info_zong = nk.ecg_peaks(ecg_signal, sampling_rate=fs, method="zong2003")
    peaks_zong = np.array(info_zong["ECG_R_Peaks"])
    
    tolerance_samples = int((tolerance_ms / 1000.0) * fs)
    agreed_peaks = []
    matched_in_alg2 = set()
    
    for p1 in peaks_hamilton:
        matches = np.where((peaks_zong >= p1 - tolerance_samples) & 
                           (peaks_zong <= p1 + tolerance_samples))[0]
        for m in matches:
            if m not in matched_in_alg2:
                agreed_peaks.append(p1) 
                matched_in_alg2.add(m)
                break
    
    agreed_count = len(agreed_peaks)
    total_unique = len(peaks_hamilton) + len(peaks_zong) - agreed_count
    bsqi_score = agreed_count / total_unique if total_unique > 0 else 0
    
    t = np.arange(len(ecg_signal)) / fs
    duration = len(ecg_signal) / fs
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    plt.subplots_adjust(hspace=0.3)
    
    ax1.plot(t, ecg_signal, color='gray', alpha=0.4, label='ECG Signal')
    ax1.scatter(peaks_hamilton/fs, ecg_signal[peaks_hamilton], 
                color='red', marker='o', s=80, label='Hamilton Peaks (Alg 1)', zorder=3)
    ax1.scatter(peaks_zong/fs, ecg_signal[peaks_zong], 
                color='blue', marker='x', s=80, label='Zong Peaks (Alg 2)', zorder=3)
    ax1.set_title(f"Step 1: Peak Detection from Two Different Algorithms\nRecord: {record_name}")
    ax1.legend(loc='upper right')
    ax1.set_ylabel("Amplitude")

    ax2.plot(t, ecg_signal, color='black', linewidth=1, alpha=0.6)
    
    first_label = True
    for p in peaks_hamilton:
        ax2.axvspan((p - tolerance_samples)/fs, (p + tolerance_samples)/fs, 
                    color='yellow', alpha=0.2, label='Tolerance Window' if first_label else "")
        first_label = False
        
    agreed_peaks = np.array(agreed_peaks)
    ax2.scatter(agreed_peaks/fs, ecg_signal[agreed_peaks], 
                color='green', marker='P', s=120, label='Agreed Peaks (Matched)', zorder=5)
    
    missed_hamilton = [p for p in peaks_hamilton if p not in agreed_peaks]
    ax2.scatter(np.array(missed_hamilton)/fs, ecg_signal[missed_hamilton], 
                color='orange', marker='v', s=100, label='Unmatched Alg 1', zorder=4)

    ax2.set_title(f"Step 2: Matching with Tolerance $\gamma = {tolerance_ms}ms$\nbSQI = {agreed_count} / ({len(peaks_hamilton)} + {len(peaks_zong)} - {agreed_count}) = {bsqi_score:.4f}")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Amplitude")
    ax2.legend(loc='upper right')
    
    zoom_start = max(0, duration/2 - 2.5)
    ax2.set_xlim(zoom_start, zoom_start + 5)
    
    plt.show()
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
    
    return M_smoothed

def calculate_ppg_sqi(ppg_signal, fs, method='mean', threshold=0.5):
    N = len(ppg_signal)
    smoothing_window = 3 * fs
    template_length = 1 * fs
    signal_centered = ppg_signal - np.mean(ppg_signal)
    
    zero_crossings = np.where(np.diff(np.sign(signal_centered)))[0]
    
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
        sqi_value = np.mean(M_smoothed)
        
    elif method == 'threshold':
        sqi_value = np.sum(M_smoothed > threshold) / N
        
    else:
        sqi_value = np.mean(M_smoothed)
        
    return float(sqi_value)


def visualize_ppg_sqi_steps(ppg_signal, fs, record_name="Unknown"):
   
    N = len(ppg_signal)
    t = np.arange(N) / fs
    signal_centered = ppg_signal - np.mean(ppg_signal)
    zero_crossings = np.where(np.diff(np.sign(signal_centered)))[0]
    
    fig = plt.figure(figsize=(15, 12))
    plt.suptitle(f"Record: {record_name}", 
                 fontsize=16, fontweight='bold', y=0.95)

    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(t, signal_centered, label='Centered PPG', color='black', alpha=0.7)
    ax1.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax1.scatter(t[zero_crossings], signal_centered[zero_crossings], color='red', s=25, label='Zero Crossings')
    ax1.set_title("Step 1 & 2: Signal Centering and Zero-Crossing Detection", loc='left')
    ax1.set_ylabel("Amplitude")
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.6)

    template_length = 1 * fs
    S_cc, S_cx = [], []
    segments = []
    
    for i in range(len(zero_crossings) - 1):
        start, end = zero_crossings[i], zero_crossings[i+1]
        seg_data = signal_centered[start:end]
        resampled_seg = resample(seg_data, template_length)
        
        seg_type = 'cc' if np.mean(seg_data) < 0 else 'cx'
        if seg_type == 'cc': S_cc.append(resampled_seg)
        else: S_cx.append(resampled_seg)
        segments.append({'start': start, 'end': end, 'resampled': resampled_seg, 'type': seg_type})

    T_cc = np.mean(S_cc, axis=0) if S_cc else np.zeros(template_length)
    T_cx = np.mean(S_cx, axis=0) if S_cx else np.zeros(template_length)

    ax2 = plt.subplot(3, 2, 3)
    for s in S_cc: ax2.plot(s, color='blue', alpha=0.1)
    ax2.plot(T_cc, color='blue', linewidth=3, label='Concave Template')
    ax2.set_title("Step 3: Concave Templates (cc)")
    ax2.legend()

    ax3 = plt.subplot(3, 2, 4)
    for s in S_cx: ax3.plot(s, color='orange', alpha=0.1)
    ax3.plot(T_cx, color='orange', linewidth=3, label='Convex Template')
    ax3.set_title("Step 4: Convex Templates (cx)")
    ax3.legend()

    M = np.zeros(N)
    for seg in segments:
        T_prime = T_cc if seg['type'] == 'cc' else T_cx
        if np.std(seg['resampled']) > 0 and np.std(T_prime) > 0:
            corr = np.corrcoef(seg['resampled'], T_prime)[0, 1]
            M[seg['start']:seg['end']] = corr

    M_norm = (M - np.min(M)) / (np.max(M) - np.min(M)) if np.max(M) > np.min(M) else M
    
    overall_sqi = np.mean(M_norm)

    ax4 = plt.subplot(3, 1, 3)
    ax4.plot(t, M_norm, label='Raw Quality Mask', color='darkgreen', linewidth=2)
    ax4.fill_between(t, 0, M_norm, color='green', alpha=0.15)
    
    ax4.axhline(overall_sqi, color='blue', linestyle='--', linewidth=1.5, 
                label=f'Global Average SQI ({overall_sqi:.4f})')
    
    ax4.axhline(0.5, color='red', linestyle=':', label='Threshold 0.5')
    
    ax4.set_title(f"Final Step: Raw Quality Estimation (Mean SQI: {overall_sqi:.4f})", loc='left')
    ax4.set_xlabel("Time (seconds)")
    ax4.set_ylabel("Quality Score")
    ax4.set_ylim(0, 1.1)
    ax4.legend(loc='upper right')
    ax4.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.show()

if __name__ == "__main__":
    path = "datasets/total_z_test.npz"
    data = np.load(path, allow_pickle=True)
    ecg_signal = data['ecgs'][2]
    ppg_signal = data['ppgs'][2]  
    record_name = data['records'][2]
    # visualize_ppg_sqi_steps(ppg_signal, fs=125, record_name=record_name)
    visualize_bsqi_steps(ecg_signal=ecg_signal, fs=125, record_name=record_name)