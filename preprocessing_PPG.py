import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline
from scipy.signal import medfilt
import os
import random
import wfdb
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from matplotlib.widgets import Slider
from scipy.stats import kurtosis
fs = 125 
SEED = 42

def set_seed(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    print(f"Random seed set to: {seed_value}")

def _butter_lowpass(data: np.ndarray, cutoff: float=10, order: int = 5) -> np.ndarray:
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        y = signal.filtfilt(b, a, data)
        return y

# def normalize_signal(signal_data: np.ndarray) -> np.ndarray:
#         min_val = 0
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

def _dc_component(time: np.ndarray, signal_data: np.ndarray) -> np.ndarray:

        inverted_signal = -1 * signal_data
        prominence_est = _get_prominence_threshold(inverted_signal)
        distance = int(0.4 * fs) 
        feet_indices, _ = signal.find_peaks(inverted_signal, distance=distance, prominence=prominence_est)
        if len(feet_indices) < 2:
            return signal_data - np.mean(signal_data)

        anchored_indices = np.concatenate(([0], feet_indices, [len(signal_data)-1]))
        feet_values = signal_data[feet_indices]
        val_start = feet_values[0]
        val_end = feet_values[-1]
        anchored_values = np.concatenate(([val_start], feet_values, [val_end]))
        dc_spline = CubicSpline(time[anchored_indices], anchored_values)
        dc_component = dc_spline(time)
        
        return dc_component

def _get_prominence_threshold(signal_data: np.ndarray) -> float:
        temp = signal_data.copy()
        temp -= np.min(temp)
        return np.median(temp) * 0.5 


def remove_large_spikes_auto(signal, sigma_factor=5):
        smoothed_signal = medfilt(signal, kernel_size=3)
        diff = np.abs(signal - smoothed_signal)
        threshold = np.mean(diff) + sigma_factor * np.std(diff)
        spike_locations = diff > threshold
        cleaned_signal = np.where(spike_locations, smoothed_signal, signal)
        return cleaned_signal


def preprocessing_PPG(ppg_signal: np.ndarray) -> np.ndarray:
        time = np.arange(len(ppg_signal)) / fs
        ppg_bpf = _butter_lowpass(ppg_signal, cutoff=10, order=5)
        ppg_normalized = normalize_signal(ppg_bpf)
        ppg_ac = ppg_normalized -  _dc_component(time, ppg_normalized)
        ppg_remove_spikes =  remove_large_spikes_auto(ppg_ac, sigma_factor=3)
        return ppg_remove_spikes


def visualize_ppg_pipeline(raw_ppg, fs, title="Optimized Pipeline"):
    
    N = len(raw_ppg)
    time = np.arange(N) / fs
    
    # --- 1: LOWPASS FILTER  ---
    ppg_lowpass = _butter_lowpass(raw_ppg, cutoff=10, order=5)
    
    # --- 2: DC REMOVAL  ---
    dc_component = _dc_component(time, ppg_lowpass)
    ppg_ac = ppg_lowpass - dc_component
    
    # --- 3: SPIKE REMOVAL  ---
    ppg_clean_ac = remove_large_spikes_auto(ppg_ac, sigma_factor=1)
    
    # --- 4: NORMALIZE  ---
    ppg_final = normalize_signal(ppg_clean_ac)
    
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
    plt.subplots_adjust(hspace=0.3)
    
    # 1. Raw
    axes[0].plot(time, raw_ppg, color='black', alpha=0.7)
    axes[0].set_title(f"1. Raw PPG Signal - {title}")
    
    # 2. Lowpass
    axes[1].plot(time, ppg_lowpass, color='green')
    axes[1].set_title("2. After Lowpass Filter (10Hz)")

    # 3. DC Removal 
    axes[2].plot(time, ppg_lowpass, color='green', alpha=0.4, label='Lowpassed Signal')
    axes[2].plot(time, dc_component, color='red', linestyle='--', linewidth=2, label='Estimated DC')
    axes[2].set_title("3. DC Removal (Done BEFORE Normalization)")
    axes[2].legend()

    # 4. Spike Removal
    axes[3].plot(time, ppg_ac, color='purple', alpha=0.5, label='Signal with Spikes')
    axes[3].plot(time, ppg_clean_ac, color='blue', linewidth=1, label='Spikes Removed')
    axes[3].set_title("4. Spike Removal (On AC Signal)")
    axes[3].legend()

    # 5. Final Normalize
    axes[4].plot(time, ppg_final, color='#d62728')
    axes[4].set_title("5. Final: Normalize z-score for Model Input")
    axes[4].set_xlabel("Time (s)")
    axes[4].grid(True, alpha=0.3)

    plt.suptitle("OPTIMIZED PPG PROCESSING PIPELINE", fontsize=16)
    plt.show()

def visualize_fft_process(raw_ppg, fs, title="FFT Analysis"):
    N = len(raw_ppg)
    time = np.arange(N) / fs
    cutoff = 10 # Hz

    # 1. Apply Lowpass Filter
    ppg_lowpass = _butter_lowpass(raw_ppg, cutoff=cutoff, order=5)
    
    # 2. Calculate FFT
    # Frequency axis
    xf = fftfreq(N, 1 / fs)[:N//2]
    
    # FFT Raw
    yf_raw = fft(raw_ppg)
    mag_raw = 2.0/N * np.abs(yf_raw[0:N//2])
    
    # FFT Filtered
    yf_filt = fft(ppg_lowpass)
    mag_filt = 2.0/N * np.abs(yf_filt[0:N//2])
    
    # 3. Plotting
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.3)
    
    # --- Plot 1: Time Domain Comparison ---
    axes[0].plot(time, raw_ppg, color='gray', alpha=0.5, label='Raw Signal (Noisy)')
    axes[0].plot(time, ppg_lowpass, color='blue', linewidth=1.5, label=f'Lowpass Filtered (Cutoff={cutoff}Hz)')
    axes[0].set_title(f"1. Time Domain Signal - {title}")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # --- Plot 2: Frequency Domain Comparison ---
    # Vẽ chồng 2 phổ lên nhau để so sánh
    axes[1].plot(xf, mag_raw, color='gray', alpha=0.5, label='Raw Spectrum')
    axes[1].plot(xf, mag_filt, color='green', linewidth=2, label='Filtered Spectrum')
    
    # Đánh dấu vùng bị cắt bỏ (> 10Hz)
    axes[1].axvspan(cutoff, fs/2, color='red', alpha=0.1, label=f'Cutoff Region (>{cutoff}Hz)')
    axes[1].axvline(cutoff, color='red', linestyle='--', linewidth=2, label='Cutoff Frequency')
    
    axes[1].set_title(f"2. Frequency Spectrum (FFT) Comparison")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Magnitude")
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # Auto-scale trục Y (bỏ qua thành phần DC tại 0Hz để zoom tốt hơn)
    if len(mag_raw) > 1:
        max_ac = np.max(mag_raw[1:]) 
        axes[1].set_ylim(0, max_ac * 1.2)

    plt.show()



def visualize_dc(raw_ppg, fs, title="DC Step"):
    """
    This function visualizes the DC Component calculation details, 
    INCLUDING the intermediate step of inverting the signal to find feet.
    """
    N = len(raw_ppg)
    time = np.arange(N) / fs
    
    # --- STEP 1: BASIC PRE-PROCESSING (Lowpass) ---
    ppg_lowpass = _butter_lowpass(raw_ppg, cutoff=10, order=5)
    
    # --- STEP 2 & 3: CALCULATE DC COMPONENT ---
    # 2a. Invert signal to find valleys (feet)
    inverted_signal = -1 * ppg_lowpass
    prominence_est = _get_prominence_threshold(inverted_signal)
    distance = int(0.4 * fs)
    
    # Find peaks on the INVERTED signal (which are valleys on the original)
    feet_indices, _ = signal.find_peaks(inverted_signal, distance=distance, prominence=prominence_est)
    
    # 2b. Create DC Curve (Spline)
    if len(feet_indices) < 2:
        dc_component = np.full_like(ppg_lowpass, np.mean(ppg_lowpass))
        anchored_indices = []
        anchored_values = []
        status_msg = "Not enough peaks for Spline -> Using Mean Subtraction"
    else:
        # Define Anchors
        anchored_indices = np.concatenate(([0], feet_indices, [len(ppg_lowpass)-1]))
        
        # Note: Using the simple logic from your snippet (taking value at index)
        anchored_values = ppg_lowpass[anchored_indices]
        feet_values = ppg_lowpass[feet_indices]
        val_start = feet_values[0]
        val_end = feet_values[-1]
        anchored_values = np.concatenate(([val_start], feet_values, [val_end]))
        dc_spline = CubicSpline(time[anchored_indices], anchored_values)
        dc_component = dc_spline(time)
        status_msg = "Using Cubic Spline to connect signal feet"

    # --- STEP 4: SUBTRACT DC ---
    ppg_ac = ppg_lowpass - dc_component

    # ==========================================
    # DETAILED VISUALIZATION
    # ==========================================
    # Change subplots to 4 rows
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    plt.subplots_adjust(hspace=0.3)
    
    # --- Plot 1: Lowpass Signal ---
    axes[0].plot(time, ppg_lowpass, color='green', alpha=0.8)
    axes[0].set_title(f"1. Signal after Lowpass Filter - {title}")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)

    # --- Plot 2: Inverted Signal (NEW STEP) ---
    # Visualizing the logic: "How did we find the feet?"
    axes[1].plot(time, inverted_signal, color='orange', alpha=0.7, label='Inverted Signal (-1 * Lowpass)')
    
    if len(feet_indices) > 0:
        # Plot the peaks found on the inverted signal
        axes[1].plot(time[feet_indices], inverted_signal[feet_indices], "x", color='black', markersize=8, markeredgewidth=2, label='Detected Peaks')
        
    axes[1].set_title("2. Intermediate Step: Inverted Signal to Detect Valleys")
    axes[1].set_ylabel("Inverted Amp")
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # --- Plot 3: DC Estimation ---
    axes[2].plot(time, ppg_lowpass, color='blue', alpha=0.4, label='Original Lowpass Signal')
    axes[2].plot(time, dc_component, color='red', linewidth=2, linestyle='--', label='Estimated DC Component')
    
    # Draw anchor points (These correspond to the peaks in Plot 2, mapped back to original signal)
    if len(anchored_indices) > 0:
        axes[2].scatter(time[anchored_indices], anchored_values, color='red', s=40, zorder=5, label='Anchor Points (Feet)')
        
    axes[2].set_title(f"3. DC Baseline Estimation ({status_msg})")
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    # --- Plot 4: Final Result ---
    axes[3].plot(time, ppg_ac, color='purple', linewidth=1.5)
    axes[3].set_title("4. Result: AC Component (Signal - DC)")
    axes[3].axhline(0, color='black', linestyle=':', alpha=0.5)
    axes[3].set_ylabel("Amplitude")
    axes[3].set_xlabel("Time (s)")
    axes[3].grid(True, alpha=0.3)

    plt.show()

def visualize_spike_removal_step(raw_ppg, fs, title="Spike Removal Analysis"):
    N = len(raw_ppg)
    time = np.arange(N) / fs
    
    # --- PRE-PROCESSING STEPS ---
    
    # 1. Lowpass Filter
    ppg_lowpass = _butter_lowpass(raw_ppg, cutoff=10, order=5)
    
    # [REMOVED] Step 2: Normalize (Đã bỏ bước này theo yêu cầu)
    # ppg_norm = normalize_signal(ppg_lowpass) 
    
    # 2. DC Removal (Thực hiện trực tiếp trên tín hiệu Lowpass)
    # Lưu ý: Hàm _dc_component phải xử lý được biên độ gốc
    dc_component = _dc_component(time, ppg_lowpass)
    
    # Tín hiệu xoay chiều (AC) giữ nguyên biên độ gốc
    ppg_ac = ppg_lowpass - dc_component
    
    # --- SPIKE REMOVAL LOGIC ---
    sigma_factor = 1
    
    # A. Create smoothed reference (Median filter)
    smoothed_signal = medfilt(ppg_ac, kernel_size=3)
    
    # B. Calculate difference
    diff = np.abs(ppg_ac - smoothed_signal)
    
    # C. Calculate Threshold
    # Vì không normalize, giá trị mean và std ở đây sẽ theo thang đo gốc
    threshold = np.mean(diff) + sigma_factor * np.std(diff)
    
    # D. Identify Spikes
    spike_locations = diff > threshold
    num_spikes = np.sum(spike_locations)
    
    # E. Replace Spikes
    cleaned_signal = np.where(spike_locations, smoothed_signal, ppg_ac)

    # ==========================================
    # DETAILED VISUALIZATION
    # ==========================================
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    plt.subplots_adjust(hspace=0.3)
    
    # --- Plot 1: Input (AC) vs Smoothed Reference ---
    axes[0].plot(time, ppg_ac, color='black', alpha=0.5, label='Input Signal (AC - Original Amp)')
    axes[0].plot(time, smoothed_signal, color='cyan', linestyle='--', linewidth=1, label='Smoothed Reference (Median)')
    axes[0].set_title(f"1. Input Signal vs Smoothed Reference - {title}")
    axes[0].set_ylabel("Original Amplitude")
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # --- Plot 2: The Math (Difference & Threshold) ---
    axes[1].plot(time, diff, color='purple', label='Difference (|Input - Smoothed|)')
    axes[1].axhline(threshold, color='red', linewidth=2, linestyle='--', label=f'Threshold (Mean + {sigma_factor}*Std)')
    
    # Highlight points above threshold
    spike_indices = np.where(spike_locations)[0]
    if len(spike_indices) > 0:
        axes[1].scatter(time[spike_indices], diff[spike_indices], color='red', zorder=5, label='Detected Spikes')
        
    axes[1].set_title(f"2. Difference Logic: Found {num_spikes} points > Threshold ({threshold:.4f})")
    axes[1].set_ylabel("Difference Magnitude")
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # --- Plot 3: Final Result (Overlay) ---
    # Draw original as ghost (gray)
    axes[2].plot(time, ppg_ac, color='gray', alpha=0.5, linewidth=3, label='Before (Original)')
    # Draw clean as solid line
    axes[2].plot(time, cleaned_signal, color='#d62728', linewidth=1.5, label='After (Cleaned)')
    
    # Mark the replaced points
    if len(spike_indices) > 0:
        axes[2].scatter(time[spike_indices], ppg_ac[spike_indices], color='orange', marker='x', s=30, linewidth=1.5, label='Removed Points', zorder=10)

    axes[2].set_title("3. Final Result: Original vs Cleaned")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Original Amplitude")
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f"Spike Removal Analysis  - {title}", fontsize=16)
    plt.show()

def survey_sigma_factors(raw_ppg, fs, title="Record", sigma_list=[1, 2, 3, 4, 5, 6]):
    """
    Plots a comparison of filtering results with different Sigma Factors.
    (WITHOUT Normalization step)
    """
    N = len(raw_ppg)
    time = np.arange(N) / fs
    
    # --- PRE-PROCESSING ---
    # 1. Lowpass Filter
    ppg_lowpass = _butter_lowpass(raw_ppg, cutoff=10, order=5)
    
    # [REMOVED] Step 2: Normalize
    # ppg_norm = normalize_signal(ppg_lowpass)
    
    # 2. DC Removal (Applied directly on Lowpass signal)
    # Lưu ý: Hàm _dc_component phải xử lý được biên độ gốc
    dc_component = _dc_component(time, ppg_lowpass)
    
    # Tín hiệu AC giữ nguyên biên độ gốc
    ppg_ac = ppg_lowpass - dc_component
    
    # Create reference data (Smoothed)
    smoothed_signal = medfilt(ppg_ac, kernel_size=3)
    diff = np.abs(ppg_ac - smoothed_signal)
    
    # Calculate stats on the original amplitude scale
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    
    # --- PLOT COMPARISON GRID ---
    num_plots = len(sigma_list)
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots), sharex=True)
    if num_plots == 1: axes = [axes] 
    
    plt.subplots_adjust(hspace=0.3)
    
    for i, sigma in enumerate(sigma_list):
        # 1. Recalculate Threshold based on this Sigma
        threshold = mean_diff + sigma * std_diff
        
        # 2. Find spikes
        spike_locations = diff > threshold
        cleaned_signal = np.where(spike_locations, smoothed_signal, ppg_ac)
        num_spikes = np.sum(spike_locations)
        
        # 3. Plot
        ax = axes[i]
        # Plot faint original signal
        ax.plot(time, ppg_ac, color='gray', alpha=0.4, label='Original (AC)')
        # Plot clean signal boldly
        ax.plot(time, cleaned_signal, color='blue', linewidth=1.5, label=f'Cleaned (Sigma={sigma})')
        
        # Mark removed points
        removed_indices = np.where(spike_locations)[0]
        if len(removed_indices) > 0:
            ax.scatter(time[removed_indices], ppg_ac[removed_indices], color='red', marker='x', s=30, label='Removed Spikes')
            
        ax.set_title(f"Sigma Factor = {sigma} | Threshold = {threshold:.4f} | Removed {num_spikes} points")
        ax.set_ylabel("Original Amplitude")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        # Warning if removing too much
        if num_spikes > N * 0.05: 
            # Vị trí text tự động theo max biên độ gốc
            ax.text(time[0], np.max(ppg_ac), "WARNING: Cutting too much!", color='red', fontweight='bold')

    plt.xlabel("Time (s)")
    plt.suptitle(f"COMPARISON OF SIGMA FACTORS (NO NORM) - {title}", fontsize=16)
    plt.show()


def interactive_sigma_tuner(raw_ppg, fs, title="Record"):
    """
    Interactive tool to tune the Sigma Factor for spike removal.
    """
    N = len(raw_ppg)
    time = np.arange(N) / fs
    
    # --- PRE-PROCESSING ---
    ppg_lowpass = _butter_lowpass(raw_ppg, cutoff=10, order=5)
    ppg_norm = normalize_signal(ppg_lowpass)
    dc_component = _dc_component(time, ppg_norm)
    ppg_ac = ppg_norm - dc_component
    
    # Create smoothed reference for calculation
    smoothed_signal = medfilt(ppg_ac, kernel_size=3)
    diff = np.abs(ppg_ac - smoothed_signal)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    
    # --- SETUP PLOT ---
    fig, (ax_diff, ax_clean) = plt.subplots(2, 1, figsize=(12, 10))
    # Reserve space for the slider at the bottom and title at the top
    plt.subplots_adjust(bottom=0.15, hspace=0.3, top=0.92) 
    
    # Add Main Title containing the Record Name
    plt.suptitle(f"Interactive Sigma Tuner - {title}", fontsize=16)
    
    # Plot 1: Difference & Threshold
    l_diff, = ax_diff.plot(time, diff, color='purple', alpha=0.6, label='Diff')
    l_thresh = ax_diff.axhline(0, color='red', linestyle='--', linewidth=2, label='Threshold') # Placeholder
    scat_spikes = ax_diff.scatter([], [], color='red', s=30, zorder=5) # Placeholder for red dots
    ax_diff.set_title("Difference vs Threshold")
    ax_diff.grid(True, alpha=0.3)
    ax_diff.legend()
    
    # Plot 2: Result
    ax_clean.plot(time, ppg_ac, color='gray', alpha=0.5, label='Original')
    l_clean, = ax_clean.plot(time, ppg_ac, color='blue', linewidth=1.5, label='Cleaned')
    scat_removed = ax_clean.scatter([], [], color='orange', marker='x', s=40, label='Removed')
    title_text = ax_clean.set_title("Result")
    ax_clean.grid(True, alpha=0.3)
    ax_clean.legend()

    # --- SLIDER SETUP ---
    # Slider position [left, bottom, width, height]
    ax_sigma = plt.axes([0.2, 0.05, 0.6, 0.03]) 
    slider = Slider(
        ax=ax_sigma,
        label='Sigma Factor',
        valmin=0.5,
        valmax=10.0,
        valinit=3.0,
        valstep=0.1
    )

    # --- UPDATE FUNCTION ---
    def update(val):
        sigma = slider.val
        
        # 1. Recalculate
        threshold = mean_diff + sigma * std_diff
        spike_mask = diff > threshold
        cleaned_signal = np.where(spike_mask, smoothed_signal, ppg_ac)
        
        spike_indices = np.where(spike_mask)[0]
        
        # 2. Update Diff plot
        l_thresh.set_ydata([threshold]) # Move red line
        if len(spike_indices) > 0:
            scat_spikes.set_offsets(np.c_[time[spike_indices], diff[spike_indices]])
        else:
            scat_spikes.set_offsets(np.zeros((0, 2)))
            
        # 3. Update Result plot
        l_clean.set_ydata(cleaned_signal)
        if len(spike_indices) > 0:
            scat_removed.set_offsets(np.c_[time[spike_indices], ppg_ac[spike_indices]])
        else:
            scat_removed.set_offsets(np.zeros((0, 2)))
            
        title_text.set_text(f"Result: Removed {len(spike_indices)} points (Sigma={sigma:.1f})")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    
    # Call update initially to initialize the plot
    update(3.0) 
    plt.show()


def optimize_sigma_with_kurtosis(raw_ppg, fs, title="Optimization"):
    N = len(raw_ppg)
    time = np.arange(N) / fs
    
    # --- 1. CONSISTENT PRE-PROCESSING (No Normalization) ---
    # Lowpass Filter
    ppg_lowpass = _butter_lowpass(raw_ppg, cutoff=10, order=5)
    
    # [REMOVED] Normalization
    # ppg_norm = normalize_signal(ppg_lowpass)
    
    # DC Removal (Applied directly on Lowpass signal with Original Amplitude)
    # Note: Ensure _dc_component handles original amplitude correctly
    dc_component = _dc_component(time, ppg_lowpass) 
    
    # AC Signal (Keeps original amplitude scale)
    ppg_ac = ppg_lowpass - dc_component
    
    # Prepare reference signals once (don't recalculate inside loop)
    smoothed_ref = medfilt(ppg_ac, kernel_size=3)
    diff_signal = np.abs(ppg_ac - smoothed_ref)
    
    # These stats are now on the Original Amplitude scale
    mean_diff = np.mean(diff_signal)
    std_diff = np.std(diff_signal)
    
    # --- 2. GRID SEARCH SIGMA ---
    sigma_candidates = np.linspace(1.0, 8.0, 20) # Test range
    kurtosis_values = []
    retained_energy = []
    
    for sigma in sigma_candidates:
        # Calculate threshold for this specific sigma
        threshold = mean_diff + sigma * std_diff
        
        # Apply logic
        spike_mask = diff_signal > threshold
        cleaned_signal = np.where(spike_mask, smoothed_ref, ppg_ac)
        
        # Metric A: Kurtosis (Lower is generally better, means less outliers)
        # Kurtosis is scale-invariant, so it works fine without normalization
        k = kurtosis(cleaned_signal)
        kurtosis_values.append(k)
        
        # Metric B: Retained Energy (Higher is better)
        # We normalize it relative to the original AC signal energy
        original_energy = np.sum(ppg_ac**2)
        current_energy = np.sum(cleaned_signal**2)
        
        if original_energy == 0:
            energy_ratio = 0
        else:
            energy_ratio = (current_energy / original_energy) * 100
            
        retained_energy.append(energy_ratio)

    # --- 3. DUAL-AXIS VISUALIZATION ---
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot Kurtosis (Red)
    color_k = 'tab:red'
    ax1.set_xlabel('Sigma Factor')
    ax1.set_ylabel('Kurtosis (Spikiness)', color=color_k, fontweight='bold')
    ax1.plot(sigma_candidates, kurtosis_values, color=color_k, marker='o', linewidth=2, label='Kurtosis')
    ax1.tick_params(axis='y', labelcolor=color_k)
    ax1.grid(True, alpha=0.3)

    # Plot Retained Energy (Blue) on the right axis
    ax2 = ax1.twinx() 
    color_e = 'tab:blue'
    ax2.set_ylabel('Retained Signal Energy (%)', color=color_e, fontweight='bold')
    ax2.plot(sigma_candidates, retained_energy, color=color_e, marker='s', linestyle='--', linewidth=1.5, label='Energy')
    ax2.tick_params(axis='y', labelcolor=color_e)

    # Title
    plt.title(f"{title}", fontsize=14)
    
    # --- 4. SUGGEST OPTIMAL POINT ---
    # Heuristic: Find where Kurtosis stabilizes
    deltas = np.diff(kurtosis_values)
    # If change is less than 0.5, we consider it "flat enough"
    stabilized_indices = np.where(np.abs(deltas) < 0.5)[0]
    
    if len(stabilized_indices) > 0:
        # Pick the first sigma where it starts to stabilize
        best_idx = stabilized_indices[0]
        best_sigma = sigma_candidates[best_idx]
        
        # Annotate
        
        ax1.annotate(f'Suggested Sigma ~ {best_sigma:.1f}\n(Elbow Point)', 
                     xy=(best_sigma, kurtosis_values[best_idx]), 
                     xytext=(best_sigma + 0.5, kurtosis_values[best_idx] + 1),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
        
        # Draw vertical line
        ax1.axvline(best_sigma, color='green', linestyle=':', alpha=0.6)

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    set_seed(SEED)
    datapath = "/home/linhhima/Pre_processing_data/Datasets/mimic_perform_non_af_wfdb" 
    record_list = ['mimic_perform_non_af_001', 'mimic_perform_non_af_002', 
                   'mimic_perform_non_af_013', 'mimic_perform_non_af_016']
    # record_list = ['mimic_perform_non_af_001', 'mimic_perform_non_af_002',
    #                 'mimic_perform_non_af_003', 'mimic_perform_non_af_004',
    #                 'mimic_perform_non_af_005', 'mimic_perform_non_af_006',
    #                 'mimic_perform_non_af_007', 'mimic_perform_non_af_008',
    #                 'mimic_perform_non_af_009', 'mimic_perform_non_af_010',
    #                 'mimic_perform_non_af_011', 'mimic_perform_non_af_012',
    #                 'mimic_perform_non_af_013', 'mimic_perform_non_af_014',
    #                 'mimic_perform_non_af_015', 'mimic_perform_non_af_016']
    # record_list = [ 'mimic_perform_non_af_002',
    #                 'mimic_perform_non_af_003', 'mimic_perform_non_af_004',
    #                 'mimic_perform_non_af_005', 
    #                 'mimic_perform_non_af_007', 'mimic_perform_non_af_008',
    #                 'mimic_perform_non_af_009', 
    #                 'mimic_perform_non_af_011', 
    #                 'mimic_perform_non_af_013', 
    #                 'mimic_perform_non_af_015', 'mimic_perform_non_af_016']
    for i, record_name in enumerate(record_list):
        print(f"\n===  {i+1}/{len(record_list)}: {record_name} ===")
        record = wfdb.rdrecord(os.path.join(datapath, record_name))
        signal_data = record.p_signal
        if signal_data is None or signal_data.shape[1] < 2:
            print(f"error {record_name} not enough PPG ECG")
            continue
            
        ppg = signal_data[:, 0]
        samples_to_plot = 10 * fs 
        visualize_ppg_pipeline(ppg[:samples_to_plot], fs, title=record_name)
        # visualize_fft_process(ppg[:samples_to_plot], fs, title=record_name)
        # visualize_dc(ppg[:samples_to_plot], fs, title=record_name)
        # visualize_spike_removal_step(ppg[:samples_to_plot], fs, title=record_name)
        # survey_sigma_factors(ppg[:samples_to_plot], fs, title=record_name)
        # interactive_sigma_tuner(ppg[:samples_to_plot], fs, title=record_name)
        # optimize_sigma_with_kurtosis(ppg[:samples_to_plot], fs, title=record_name)


        
