import numpy as np
from scipy import signal
from scipy.signal import medfilt
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import random
import os
import wfdb
from scipy.stats import kurtosis
SEED = 42

def set_seed(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    print(f"Random seed set to: {seed_value}")

def butter_bandpass(data: np.ndarray, lowcut: float, highcut: float, order: int = 5, fs: float=125) -> np.ndarray:
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq 
        low = np.clip(low, 0, 0.99)
        high = np.clip(high, 0, 0.99)
        
        b, a = signal.butter(order, [low, high], btype="band")
        y = signal.filtfilt(b, a, data)
        return y

def notch_filter_ecg(data: np.ndarray, notch_freq: float = 50.0, Q: float = 30.0, fs: float=125) -> np.ndarray:
        nyq = 0.5 * fs
        w0 = notch_freq / nyq
        if w0 >= 1.0:
             return data
        b, a = signal.iirnotch(w0, Q)
        y = signal.filtfilt(b, a, data)
        return y

def remove_large_spikes_auto(signal, sigma_factor=5):
        smoothed_signal = medfilt(signal, kernel_size=3)
        diff = np.abs(signal - smoothed_signal)
        threshold = np.mean(diff) + sigma_factor * np.std(diff)
        spike_locations = diff > threshold
        cleaned_signal = np.where(spike_locations, smoothed_signal, signal)
        return cleaned_signal

# def normalize_signal(signal_data: np.ndarray) -> np.ndarray:
#         min_val = np.min(signal_data)
#         max_val = np.max(signal_data)
#         if max_val - min_val < 1e-8:
#             return np.zeros_like(signal_data)
#         return (signal_data - min_val) / (max_val - min_val)

def normalize_signal(signal_data: np.ndarray) -> np.ndarray:
    mean_val = np.mean(signal_data)
    std_val = np.std(signal_data)
    if std_val < 1e-8:
        return np.zeros_like(signal_data)
        
    return (signal_data - mean_val) / std_val

def preprocessing_ECG(ecg_signal: np.ndarray, fs: float=125) -> np.ndarray:
        filtered_bandpass = butter_bandpass(ecg_signal, lowcut=0.5, highcut=100.0, order=5, fs=fs)
        filtered_notch = notch_filter_ecg(filtered_bandpass, notch_freq=50, Q=30, fs=fs)
        ecg_normalized = normalize_signal(filtered_notch)
        return ecg_normalized

def visualize_fft_bandpass(raw_data, fs, lowcut, highcut, title="Bandpass FFT Analysis"):
    N = len(raw_data)
    time = np.arange(N) / fs
    filtered_data = butter_bandpass(raw_data, lowcut, highcut, order=4, fs=fs)

    # 2. Tính toán FFT
    # Trục tần số (chỉ lấy nửa dương)
    xf = fftfreq(N, 1 / fs)[:N//2]
    
    # FFT tín hiệu gốc
    yf_raw = fft(raw_data)
    mag_raw = 2.0/N * np.abs(yf_raw[0:N//2])
    
    # FFT tín hiệu đã lọc
    yf_filt = fft(filtered_data)
    mag_filt = 2.0/N * np.abs(yf_filt[0:N//2])

    # 3. Vẽ đồ thị
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.3)

    # --- Plot 1: Miền thời gian (Time Domain) ---
    axes[0].plot(time, raw_data, color='gray', alpha=0.5, label='Raw Signal (Mixed)')
    axes[0].plot(time, filtered_data, color='blue', linewidth=1.5, label=f'Filtered ({lowcut}-{highcut} Hz)')
    axes[0].set_title(f"1. Time Domain Signal - {title}")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # --- Plot 2: Miền tần số (Frequency Domain) ---
    axes[1].plot(xf, mag_raw, color='gray', alpha=0.5, label='Raw Spectrum')
    axes[1].plot(xf, mag_filt, color='green', linewidth=2, label='Filtered Spectrum')
    
    # Vẽ vùng Passband (Vùng tần số giữ lại)
    axes[1].axvspan(lowcut, highcut, color='green', alpha=0.1, label='Passband Region')
    axes[1].axvline(lowcut, color='red', linestyle='--', alpha=0.7)
    axes[1].axvline(highcut, color='red', linestyle='--', alpha=0.7)
    
    axes[1].set_title(f"2. Frequency Spectrum (FFT) - Passband: [{lowcut}Hz - {highcut}Hz]")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Magnitude")
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # Auto-scale trục Y (bỏ qua thành phần DC tại 0Hz nếu nó quá lớn)
    if len(mag_raw) > 1:
        # Tìm max trong vùng quan tâm để zoom biểu đồ cho đẹp
        max_val = np.max(mag_raw[1:]) 
        axes[1].set_ylim(0, max_val * 1.2)

    plt.show()



def visualize_notch_filter(raw_data, fs, notch_freq=60.0, Q=30.0, title="Notch Filter Analysis"):
    """
    Visualizes the effect of a Notch Filter in both Time and Frequency domains.
    """
    N = len(raw_data)
    time = np.arange(N) / fs

    # 1. Apply Bandpass Filter
    filtered_data = butter_bandpass(raw_data, lowcut=0.5, highcut=100.0, order=5, fs=fs)
    # 2. Apply Notch Filter on the Bandpassed data
    notch_filtered_data = notch_filter_ecg(filtered_data, notch_freq=notch_freq, Q=Q, fs=fs)

    # --- FFT CALCULATION ---
    xf = fftfreq(N, 1 / fs)[:N//2]
    
    # FFT of the signal BEFORE Notch (but after Bandpass)
    yf_raw = fft(filtered_data)
    mag_raw = 2.0/N * np.abs(yf_raw[0:N//2])
    
    # FFT of the signal AFTER Notch
    yf_filt = fft(notch_filtered_data)
    mag_filt = 2.0/N * np.abs(yf_filt[0:N//2])

    # --- PLOTTING ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.3)

    # Plot 1: Time Domain
    # Màu xám: Tín hiệu còn nhiễu 50Hz. Màu đỏ: Tín hiệu đã sạch.
    axes[0].plot(time, filtered_data, color='gray', alpha=0.6, label='Input to Notch (Bandpass Only)')
    axes[0].plot(time, notch_filtered_data, color='#d62728', linewidth=1.5, label='Output (Notch Filtered)')
    axes[0].set_title(f"1. Time Domain Signal - {title}")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Frequency Domain
    # Màu xám: Có cái gai lớn ở 50Hz. Màu xanh: Cái gai đó bị lõm xuống.
    axes[1].plot(xf, mag_raw, color='gray', alpha=0.6, label='Spectrum (Bandpass Only)')
    axes[1].plot(xf, mag_filt, color='blue', linewidth=1.5, label='Spectrum (Notch Filtered)')

    # Highlight Notch Frequency
    axes[1].axvline(notch_freq, color='red', linestyle='--', linewidth=2, label=f'Notch Freq ({notch_freq}Hz)')
    
    # Highlight Attenuation Bandwidth
    bandwidth = notch_freq / Q
    axes[1].axvspan(notch_freq - bandwidth, notch_freq + bandwidth, color='red', alpha=0.1, label='Attenuated Band')

    axes[1].set_title(f"2. Frequency Spectrum (FFT) - Removal at {notch_freq}Hz (Q={Q})")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Magnitude")
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    # Auto-scale Y-axis (bỏ qua thành phần DC để zoom tốt hơn)
    if len(mag_raw) > 1:
        max_val = np.max(mag_raw[1:])
        axes[1].set_ylim(0, max_val * 1.2)

    plt.show()

def visualize_spike_removal(raw_ecg, fs, sigma_factor=4, title="Spike Removal Analysis"):
    """
    Visualizes the internal steps of the remove_large_spikes_auto function.
    Input: 'filtered_ppg' is assumed to already be Bandpassed and Notched.
    """
    N = len(raw_ecg)
    time = np.arange(N) / fs
    filtered_data = butter_bandpass(raw_ecg, lowcut=0.5, highcut=100.0, order=5, fs=fs)
    
    # 2. Notch Filter (e.g., 60Hz or 50Hz)
    notch_filtered_data = notch_filter_ecg(filtered_data, notch_freq=60.0, Q=30.0, fs=fs)

    # --- PHASE 2: SPIKE REMOVAL LOGIC REPLICATION ---
    # Step 1: Smoothed Reference (Median Filter)
    smoothed_ref = medfilt(notch_filtered_data, kernel_size=3)

    # Step 2: Difference Calculation
    diff_signal = np.abs(notch_filtered_data - smoothed_ref)

    # Step 3: Threshold Calculation
    mean_diff = np.mean(diff_signal)
    std_diff = np.std(diff_signal)
    threshold_val = mean_diff + sigma_factor * std_diff
    
    # Step 4: Identify Spikes (Mask)
    spike_mask = diff_signal > threshold_val
    spike_indices = np.where(spike_mask)[0]
    num_spikes = len(spike_indices)
    
    # Step 5: Final Replacement (Using the actual function)
    final_cleaned_signal = remove_large_spikes_auto(notch_filtered_data, sigma_factor)

    # --- PHASE 3: PLOTTING ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
    plt.subplots_adjust(top=0.92, hspace=0.3)
    
    # Tiêu đề chính bao gồm cả thông số Sigma
    fig.suptitle(f"ECG Spike Removal Process (Sigma={sigma_factor}) - {title}", fontsize=16, y=0.99)
    
    # Plot 1: Pre-processed Input vs Smoothed Reference
    axes[0].plot(time, notch_filtered_data, color='blue', alpha=0.6, label='Input (Bandpass + Notch)')
    axes[0].plot(time, smoothed_ref, color='orange', linewidth=1.5, label='Reference (Median Filter)')
    axes[0].set_title(f"1. Input Signal vs Smoothed Reference")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Detection Logic
    axes[1].plot(time, diff_signal, color='purple', alpha=0.7, label='Difference |Input - Ref|')
    
    label_thresh = f'Threshold ({mean_diff:.2f} + {sigma_factor}*{std_diff:.2f}) = {threshold_val:.2f}'
    axes[1].axhline(threshold_val, color='red', linewidth=2, linestyle='--', label=label_thresh)
    
    if num_spikes > 0:
        axes[1].scatter(time[spike_indices], diff_signal[spike_indices], 
                        color='red', s=50, zorder=5, marker='o', label=f'Detected Spikes ({num_spikes})')
        
    axes[1].set_title("2. Detection Logic: Difference vs Threshold")
    axes[1].set_ylabel("Diff Magnitude")
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Final Comparison
    axes[2].plot(time, notch_filtered_data, color='gray', alpha=0.5, linewidth=2, label='Before (With Spikes)')
    axes[2].plot(time, final_cleaned_signal, color='#d62728', linewidth=1.5, label='After (Cleaned)')
    
    if num_spikes > 0:
        axes[2].scatter(time[spike_indices], notch_filtered_data[spike_indices], 
                        color='orange', marker='x', s=100, linewidth=2, label='Removed Points', zorder=10)

    axes[2].set_title("3. Final Result Comparison")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Amplitude")
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    plt.show()

def find_optimal_sigma(raw_ecg, fs, title="Sigma Optimization"):
    N = len(raw_ecg)
    time = np.arange(N) / fs

    # --- PHASE 1: PRE-PROCESSING (Run Once) ---
    # 1. Bandpass Filter (0.5 - 100Hz)
    filtered_data = butter_bandpass(raw_ecg, lowcut=0.5, highcut=100.0, order=5, fs=fs)
    
    # 2. Notch Filter (60Hz)
    notch_filtered_data = notch_filter_ecg(filtered_data, notch_freq=60.0, Q=30.0, fs=fs)
    
    # --- PHASE 2: PREPARE SPIKE LOGIC ---
    # We calculate the reference signal ONCE to save speed
    smoothed_ref = medfilt(notch_filtered_data, kernel_size=3)
    diff_signal = np.abs(notch_filtered_data - smoothed_ref)
    
    mean_diff = np.mean(diff_signal)
    std_diff = np.std(diff_signal)
    
    # Calculate Energy of the input (Reference for ratio)
    original_energy = np.sum(notch_filtered_data**2)

    # --- PHASE 3: GRID SEARCH ---
    sigma_candidates = np.linspace(1.0, 8.0, 20) # Test Sigmas from 2.0 to 8.0
    kurtosis_list = []
    energy_list = []
    
    for sigma in sigma_candidates:
        # Calculate Threshold based on current sigma
        threshold_val = mean_diff + sigma * std_diff
        
        # Apply Logic
        spike_mask = diff_signal > threshold_val
        
        # Create Cleaned Signal
        # (Where spike detected -> replace with smoothed_ref, else keep original)
        cleaned_signal = np.where(spike_mask, smoothed_ref, notch_filtered_data)
        
        # Metric A: Kurtosis (We want this LOW -> less spiky)
        k = kurtosis(cleaned_signal)
        kurtosis_list.append(k)
        
        # Metric B: Retained Energy (We want this HIGH -> keep signal)
        current_energy = np.sum(cleaned_signal**2)
        ratio = (current_energy / original_energy)
        energy_list.append(ratio)

    # --- PHASE 4: SELECT BEST SIGMA (Distance Method) ---
    K = np.array(kurtosis_list)
    E = np.array(energy_list)
    
    # Normalize metrics to 0-1 range to compare them fairly
    K_norm = (K - K.min()) / (K.max() - K.min()) if K.max() != K.min() else K
    E_norm = (E - E.min()) / (E.max() - E.min()) if E.max() != E.min() else E
    
    # Calculate Distance to "Ideal Point"
    # Ideal: Kurtosis is Minimum (Norm=0) AND Energy is Maximum (Norm=1)
    # Distance = sqrt( (K_norm - 0)^2 + (E_norm - 1)^2 )
    dist = np.sqrt(K_norm**2 + (1 - E_norm)**2)
    
    best_idx = np.argmin(dist)
    best_sigma = sigma_candidates[best_idx]
    
    # --- VISUALIZATION ---
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot Kurtosis
    color = 'tab:red'
    ax1.set_xlabel('Sigma Factor')
    ax1.set_ylabel('Kurtosis (Spikiness)', color=color, fontweight='bold')
    ax1.plot(sigma_candidates, kurtosis_list, color=color, linewidth=2, label='Kurtosis (Lower is Better)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    # Plot Energy
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Retained Energy Ratio', color=color, fontweight='bold')
    ax2.plot(sigma_candidates, energy_list, color=color, linestyle='--', linewidth=2, label='Energy (Higher is Better)')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Highlight Winner
    ax1.axvline(best_sigma, color='green', linestyle=':', linewidth=2)
    
    # Annotation
    stats = (f"OPTIMAL SIGMA: {best_sigma:.2f}\n"
             f"Energy: {energy_list[best_idx]*100:.1f}%\n"
             f"Kurtosis: {kurtosis_list[best_idx]:.2f}")
    
    bbox = dict(boxstyle="round", fc="white", ec="green", alpha=0.9)
    ax1.text(best_sigma, (np.max(kurtosis_list) + np.min(kurtosis_list))/2, stats, 
             bbox=bbox, ha='center', color='green', fontweight='bold')
    
    plt.title(f"Optimization: Trade-off between Signal Quality and Spike Removal - {title}")
    plt.tight_layout()
    plt.show()
    
    return best_sigma

def visualize_preprocessing_steps(raw_ecg: np.ndarray, fs: float, title: str = "Sigma Optimization"):
  
    # 1. Bandpass: 0.5 - 100 Hz
    signal_bandpass = butter_bandpass(raw_ecg, lowcut=0.5, highcut=100.0, order=5, fs=fs)
    
    # 2. Notch: 50 Hz (Thay đổi thành 60 nếu ở Mỹ/Hàn/Đài Loan...)
    signal_notch = notch_filter_ecg(signal_bandpass, notch_freq=60.0, Q=30.0, fs=fs)
    
    # 3. Normalize: Z-score
    signal_norm = normalize_signal(signal_notch)
    
    # --- BƯỚC 2: VẼ BIỂU ĐỒ ---
    # Tạo trục thời gian (giây)
    time_axis = np.arange(len(raw_ecg)) / fs
    
    # Setup subplot: 4 hàng, 1 cột, sharex=True để zoom đồng bộ
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    # Tiêu đề chung
    fig.suptitle(f"ECG Preprocessing Pipeline: {title}", fontsize=16, fontweight='bold', y=0.98)

    # Plot 1: Raw
    axes[0].plot(time_axis, raw_ecg, color='#333333', alpha=0.8, linewidth=1)
    axes[0].set_ylabel("Amplitude (mV)")
    axes[0].set_title("1. Raw Signal", loc='left', fontweight='bold', color='#333333')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Plot 2: Bandpass
    axes[1].plot(time_axis, signal_bandpass, color='#1f77b4', alpha=0.8, linewidth=1) # Màu xanh
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title("2. After Bandpass Filter (0.5 - 100Hz)", loc='left', fontweight='bold', color='#1f77b4')
    axes[1].grid(True, linestyle='--', alpha=0.5)

    # Plot 3: Notch
    axes[2].plot(time_axis, signal_notch, color='#2ca02c', alpha=0.8, linewidth=1) # Màu xanh lá
    axes[2].set_ylabel("Amplitude")
    axes[2].set_title("3. After Notch Filter (60Hz Removal)", loc='left', fontweight='bold', color='#2ca02c')
    axes[2].grid(True, linestyle='--', alpha=0.5)

    # Plot 4: Normalized
    axes[3].plot(time_axis, signal_norm, color='#d62728', alpha=0.9, linewidth=1.2) # Màu đỏ
    axes[3].set_ylabel("Z-Score")
    axes[3].set_xlabel("Time (seconds)")
    axes[3].set_title("4. Final Normalized Signal (Mean=0, Std=1)", loc='left', fontweight='bold', color='#d62728')
    axes[3].grid(True, linestyle='--', alpha=0.5)
    
    # Vẽ đường tham chiếu 0 cho biểu đồ cuối
    axes[3].axhline(0, color='black', linewidth=0.8, linestyle='--')

    plt.tight_layout()
    plt.subplots_adjust(top=0.93) # Chừa chỗ cho suptitle
    plt.show()

if __name__ == "__main__":
    set_seed(SEED)
    datapath = "/home/linhhima/Pre_processing_data/Datasets/mimic_perform_non_af_wfdb" 
    record_list = ['mimic_perform_non_af_001', 'mimic_perform_non_af_002', 
                   'mimic_perform_non_af_013', 'mimic_perform_non_af_016']
    fs = 125
    for i, record_name in enumerate(record_list):
        print(f"\n===  {i+1}/{len(record_list)}: {record_name} ===")
        record = wfdb.rdrecord(os.path.join(datapath, record_name))
        signal_data = record.p_signal
        if signal_data is None or signal_data.shape[1] < 2:
            print(f"error {record_name} not enough PPG ECG")
            continue
            
        ecg = signal_data[:, 1]
        samples_to_plot = 10 * fs
        # visualize_fft_bandpass(ecg[:samples_to_plot], fs=fs, lowcut=0.5, highcut=100.0, title=record_name)
        # visualize_notch_filter(ecg[:samples_to_plot], fs=fs, notch_freq=60.0, Q=30.0, title=record_name)
        # visualize_spike_removal(ecg[:samples_to_plot], fs=fs, sigma_factor=10, title=record_name)
        # find_optimal_sigma(ecg[:samples_to_plot], fs=fs, title=record_name)
        visualize_preprocessing_steps(ecg[:samples_to_plot], fs=fs, title=record_name)