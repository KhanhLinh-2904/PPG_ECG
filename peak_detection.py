import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# --- Load dataset ---
data_path = "MIMIC_val.npz"
data = np.load(data_path)

ppgs = data["ppgs"]   # shape: (N, L)
ecgs = data["ecgs"]   # shape: (N, L)
labels = data["labels"]

# --- Parameters ---
fs = 125  # sampling rate (Hz)
distance = 30  # ~0.24s apart
prominence_ppg = 0.2
prominence_ecg = 0.5




# Arrays to store peak-only signals
ppgs_peak_only = np.zeros_like(ppgs)
ecgs_peak_only = np.zeros_like(ecgs)
for i in range(len(ppgs)):
    ppg = ppgs[i]
    ecg = ecgs[i]

    # --- PPG peaks ---
    ppg_peaks, _ = find_peaks(ppg, distance=distance, prominence=prominence_ppg)  
    ppg_peak_only = np.zeros_like(ppg)
    ppg_peak_only[ppg_peaks] = ppg[ppg_peaks]

    # --- ECG peaks (QRS detection via simple peak-finding) ---
    ecg_peaks, _ = find_peaks(ecg, distance=distance, prominence=prominence_ecg)  
    ecg_peak_only = np.zeros_like(ecg)
    ecg_peak_only[ecg_peaks] = ecg[ecg_peaks]

    # Save back
    ppgs_peak_only[i] = ppg_peak_only
    ecgs_peak_only[i] = ecg_peak_only


# Save results
out_path = "MIMIC_val_peaks.npz"
np.savez(out_path, ppgs=ppgs_peak_only, ecgs=ecgs_peak_only, labels=labels)

print(f"Saved peak-only signals to {out_path}")

# --- Choose signal to visualize ---
signal_type = "ppg"   # "ppg" or "ecg"
idx = 0               # index of sample to visualize

if signal_type == "ppg":
    signal = ppgs[idx]
    peaks, _ = find_peaks(signal, distance=distance, prominence=prominence_ppg)
elif signal_type == "ecg":
    signal = ecgs[idx]
    peaks, _ = find_peaks(signal, distance=distance, prominence=prominence_ecg)
else:
    raise ValueError("signal_type must be 'ppg' or 'ecg'")

# --- Create peak-only signal ---
peak_only = np.zeros_like(signal)
peak_only[peaks] = signal[peaks]

# --- Plotting ---
time = np.arange(len(signal)) / fs

plt.figure(figsize=(12, 6))

# Original signal with peaks
plt.subplot(2, 1, 1)
plt.plot(time, signal, label=f"Original {signal_type.upper()}")
plt.plot(time[peaks], signal[peaks], "rx", label="Detected Peaks")
plt.title(f"{signal_type.upper()} Signal with Detected Peaks")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()

# Peak-only signal
plt.subplot(2, 1, 2)
plt.plot(time, peak_only, label="Peak-only Signal", color="purple")
plt.title(f"{signal_type.upper()} Peak-only Signal (Non-peaks = 0)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()

plt.tight_layout()

# Save image
out_file = f"{signal_type}_{idx}_peaks.png"
plt.savefig(out_file, dpi=300)  # high quality image
plt.show()

print(f"Figure saved as {out_file}")
