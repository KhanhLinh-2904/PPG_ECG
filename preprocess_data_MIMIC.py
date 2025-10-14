import os
import numpy as np
import wfdb
import random
from scipy.signal import butter, filtfilt, welch, find_peaks
SLICE_LENGTH = 2400  # fixed length of each slice

def isECG(ecg_segment, fs=125):
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

def load_and_slice_signals(datapath, label_value):
    """
    Load signals and slice them into fixed-length segments (2400 timesteps).
    Only keep slices of exact length 2400. Discard shorter signals.
    Returns lists: ppgs, ecgs, labels
    """
    ppgs, ecgs, labels = [], [], []

    # Only take files with .dat
    record_files = [f.split('.')[0] for f in os.listdir(datapath) if f.endswith('.dat')]

    for record_name in record_files:
        record_path = os.path.join(datapath, record_name)
        record = wfdb.rdrecord(record_path)

        # Assuming: column 0 = PPG, column 1 = ECG
        ppg_signal = record.p_signal[:, 0]
        ecg_signal = record.p_signal[:, 1]
        ecg_signal = butter_bandpass_filter(ecg_signal, 0.67, 40, 125, order=1)
        # Only consider signals that are at least 2400 long
        min_len = min(len(ppg_signal), len(ecg_signal))
        if min_len < SLICE_LENGTH:
            continue  # discard short signals

        # Slice signals into non-overlapping 2400-length chunks
        start_idx = 0
        while start_idx + SLICE_LENGTH <= min_len:
            ppg_slice = ppg_signal[start_idx:start_idx + SLICE_LENGTH]
            ecg_slice = ecg_signal[start_idx:start_idx + SLICE_LENGTH]

            # 🔍 Check NaN values
            if np.isnan(ppg_slice).any() or np.isnan(ecg_slice).any() or not isECG(ecg_slice):
                print(f"⚠️ NaN detected in record {record_name}, slice {start_idx}:{start_idx+SLICE_LENGTH} or non_ecg")
                # Option 1: skip this slice
                start_idx += SLICE_LENGTH
                continue
                
            ppg_slice = normalize_signal(ppg_slice)
            ecg_slice = normalize_signal(ecg_slice)
            ppgs.append(ppg_slice)
            ecgs.append(ecg_slice)
            labels.append(label_value)
            start_idx += SLICE_LENGTH  # move to next slice

    return ppgs, ecgs, labels


def split_and_save(ppgs, ecgs, labels, save_prefix, ratios=(0.7, 0.15, 0.15)):
    """
    Shuffle and split dataset into train/val/test.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6

    n_total = len(labels)
    indices = list(range(n_total))
    random.shuffle(indices)

    n_train = int(ratios[0] * n_total)
    n_val   = int(ratios[1] * n_total)
    n_test  = n_total - n_train - n_val

    splits = {
        "train": indices[:n_train],
        "val": indices[n_train:n_train + n_val],
        "test": indices[n_train + n_val:]
    }

    for split_name, split_idx in splits.items():
        split_ppgs   = np.array([ppgs[i] for i in split_idx], dtype=np.float32)
        split_ecgs   = np.array([ecgs[i] for i in split_idx], dtype=np.float32)
        split_labels = np.array([labels[i] for i in split_idx], dtype=np.int64)

        # save_path = f"{save_prefix}_{split_name}.npz"
        save_path = f"datasets/{save_prefix}_{split_name}.npz"

        np.savez(save_path, ecgs=split_ecgs, ppgs=split_ppgs, labels=split_labels)
        print(f"Saved {split_name} set with {len(split_labels)} samples → {save_path}")


if __name__ == "__main__":
    # Load and slice AF signals
    af_data = "datasets/mimic_perform_af_wfdb"
    ppgs_af, ecgs_af, labels_af = load_and_slice_signals(af_data, 1)

    # Load and slice non-AF signals
    non_af_data = "datasets/mimic_perform_non_af_wfdb"
    ppgs_non, ecgs_non, labels_non = load_and_slice_signals(non_af_data, 0)

    # Combine all slices
    ppgs   = ppgs_af + ppgs_non
    ecgs   = ecgs_af + ecgs_non
    labels = labels_af + labels_non

    # Split and save
    split_and_save(ppgs, ecgs, labels, save_prefix="MIMIC", ratios=(0.7, 0.15, 0.15))
