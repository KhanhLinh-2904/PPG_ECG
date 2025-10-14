import wfdb
import numpy as np
import os
from scipy.signal import butter, filtfilt, welch, find_peaks

def isECG(ecg_segment, fs=250):
    """
    Calculate the Power Spectral Density (PSD) of an ECG segment using Welch's method.

    Args:
        ecg_segment (np.ndarray): 1D array of ECG signal.
        fs (int): Sampling frequency in Hz.

    Returns:
        True: is ECG signal
        False: not ECG signal
    """
    freqs, psd = welch(ecg_segment, fs=fs, nperseg=fs*2)
    mask = (freqs >= 0)& (freqs <= 5.0)
    freqs = freqs[mask]
    psd = psd[mask]
     # --- Step 2: Find peaks ---
    peaks, _ = find_peaks(psd, height=np.mean(psd) + np.std(psd))  # strong peaks
    peak_freqs = freqs[peaks]
    peak_powers = psd[peaks]

    has_ecg = False
    if np.any((peak_freqs >= 2.5) & (peak_freqs <= 5.0)):
        has_ecg = True
    else:
        low_band_mask = (peak_freqs >= 0)& (peak_freqs <= 2.5)
        total_power = np.sum(peak_powers[low_band_mask])
        if total_power > 1e6:
            has_ecg = False
        else:
            has_ecg = True
    return has_ecg

def butter_bandpass_filter(signal, lowcut, highcut, fs, order=4):
    """
    Apply a Butterworth bandpass filter to the input signal.

    Args:
        signal (np.ndarray): 1D input signal.
        lowcut (float): Low cutoff frequency (Hz).
        highcut (float): High cutoff frequency (Hz).
        fs (int): Sampling frequency (Hz).
        order (int): Filter order.

    Returns:
        np.ndarray: Filtered signal.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(order, [low, high], btype="band")
    filtered = filtfilt(b, a, signal)
    return filtered
def normalize_signal(signal):
    """
    Normalize signal with min-max normalization to [0, 1].
    If max == min, return zeros to avoid division by zero.
    """
    min_val = np.min(signal)
    max_val = np.max(signal)
    if max_val - min_val < 1e-8:  # tránh chia cho 0
        return np.zeros_like(signal)
    return (signal - min_val) / (max_val - min_val)

def extractData(ann_atr, ecg, seg_length=2400):
    indices = ann_atr.sample       # annotation sample positions
    labels = ann_atr.aux_note      # corresponding labels
    ECG_segments = []
    segment_labels = []

    for i in range(len(indices) - 1):
        start = indices[i]
        end = indices[i + 1]

        if start < len(ecg):
            seg = ecg[start:end] if end <= len(ecg) else ecg[start:]
            seg_len = len(seg)

            # Only keep if segment has at least seg_length
            if seg_len >= seg_length:
                # Chop into fixed 2400 windows
                num_chunks = seg_len // seg_length
                for j in range(num_chunks):
                    chunk = seg[j*seg_length:(j+1)*seg_length]
                    if  np.isnan(chunk).any() or not isECG(chunk):
                        continue
                    chunk = normalize_signal(chunk)
                    ECG_segments.append(chunk)
                    if labels[i] == "(AFIB" :
                        segment_labels.append(1)
                    elif labels[i] == "(N" :
                        segment_labels.append(0)

            # else: forget it (skip)
    
    return ECG_segments, segment_labels

        
def loadData(data_path="datasets/mit-bih-atrial-fibrillation-database-1.0.0", save_path="datasets/ECG_MIT-BIH_data.npz"):
    all_segments = []
    all_labels = []

    record_files = [f.split('.')[0] for f in os.listdir(data_path) if f.endswith('.dat')]
    for record_name in record_files:
        hea_path = os.path.join(data_path, record_name + ".hea")
        atr_path = os.path.join(data_path, record_name + ".atr")
        qrs_path = os.path.join(data_path, record_name + ".qrs")
    
        # Skip if files missing
        if not os.path.exists(hea_path) or not os.path.exists(atr_path) or not os.path.exists(qrs_path):
            print(f"Skipping {record_name}: Missing .hea, .atr, or .qrs file.")
            continue  
        
        print("====================================================")
        print(f"Processing record: {record_name}")

        # Read ECG
        record1 = wfdb.rdrecord(os.path.join(data_path, record_name), channels=[0]) 
        ecg = record1.p_signal[:,0]
        Fs = record1.fs
        ecg = butter_bandpass_filter(ecg, 0.67, 40, Fs, order=1)
        ecg = normalize_signal(ecg)

        # Read annotations
        ann_atr = wfdb.rdann(os.path.join(data_path, record_name), 'atr')  

        # Segment ECG
        ECG_segments, segment_labels = extractData(ann_atr, ecg)

        # Collect
        all_segments.extend(ECG_segments)
        all_labels.extend(segment_labels)

    # Convert to arrays (labels as np.array of strings)
    all_segments = np.array(all_segments)  
    all_labels = np.array(all_labels)

    # Save to .npz
    np.savez_compressed(save_path, ecgs=all_segments, labels=all_labels)
    print(f"Saved dataset to {save_path} with {len(all_segments)} segments.")

    return all_segments, all_labels
       
      
       
    


       


# Call the function
if __name__ == "__main__":
    loadData()
    
