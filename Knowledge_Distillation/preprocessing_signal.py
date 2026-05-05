import numpy as np
from scipy import signal
from scipy.signal import medfilt
from scipy.interpolate import CubicSpline
from typing import List, Tuple, Dict, Any
import os
import wfdb
import matplotlib.pyplot as plt
import time
import random
from collections import defaultdict
class SignalProcessor:
    def __init__(self, fs: float):
        self.fs = fs

    def _butter_bandpass(self, data: np.ndarray, lowcut: float, highcut: float, order: int = 5) -> np.ndarray:
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq 
        low = np.clip(low, 0, 0.99)
        high = np.clip(high, 0, 0.99)
        
        b, a = signal.butter(order, [low, high], btype="band")
        y = signal.filtfilt(b, a, data)
        return y

    def _butter_lowpass(self, data: np.ndarray, cutoff: float=10, order: int = 5) -> np.ndarray:
        nyq = 0.5 * self.fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        y = signal.filtfilt(b, a, data)
        return y

    def _notch_filter_ecg(self, data: np.ndarray, notch_freq: float = 50.0, Q: float = 30.0) -> np.ndarray:
        nyq = 0.5 * self.fs
        w0 = notch_freq / nyq
        if w0 >= 1.0:
             print(f"⚠️ Notch filter {notch_freq}Hz is too high for FS={self.fs}Hz. Skipping.")
             return data
        b, a = signal.iirnotch(w0, Q)
        y = signal.filtfilt(b, a, data)
        return y

    def _get_prominence_threshold(self, signal_data: np.ndarray) -> float:
        temp = signal_data.copy()
        temp -= np.min(temp)
        return np.median(temp) * 0.5 

    
    def _dc_removal(self, time: np.ndarray, signal_data: np.ndarray) -> np.ndarray:
      
        inverted_signal = -1 * signal_data
        prominence_est = self._get_prominence_threshold(inverted_signal)
        distance = int(0.4 * self.fs) 
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
        ac_component = signal_data - dc_component
        
        return ac_component
 
    def remove_large_spikes_auto(self, signal, sigma_factor=5):
        smoothed_signal = medfilt(signal, kernel_size=3)
        diff = np.abs(signal - smoothed_signal)
        threshold = np.mean(diff) + sigma_factor * np.std(diff)
        spike_locations = diff > threshold
        cleaned_signal = np.where(spike_locations, smoothed_signal, signal)
        return cleaned_signal
    
    # def normalize_signal(self, signal_data: np.ndarray) -> np.ndarray:
    #     min_val = np.min(signal_data)
    #     max_val = np.max(signal_data)
    #     if max_val - min_val < 1e-8:
    #         return np.zeros_like(signal_data)
    #     return (signal_data - min_val) / (max_val - min_val)


    # def normalize_signal(self, signal_data: np.ndarray) -> np.ndarray:
    #     mean_val = np.mean(signal_data)
    #     std_val = np.std(signal_data)
        
    #     if std_val < 1e-8:
    #         return np.zeros_like(signal_data)
            
    #     z_norm_signal = (signal_data - mean_val) / std_val
        
    #     min_val = np.min(z_norm_signal)
    #     max_val = np.max(z_norm_signal)
        
    #     if max_val - min_val < 1e-8:
    #         return np.zeros_like(signal_data)
            
    #     scaled_signal = 2 * ((z_norm_signal - min_val) / (max_val - min_val)) - 1
    #     return scaled_signal
    
    def normalize_signal(self, signal_data: np.ndarray) -> np.ndarray:
        mean_val = np.mean(signal_data)
        std_val = np.std(signal_data)
        if std_val < 1e-8:
            return np.zeros_like(signal_data)
        return (signal_data - mean_val) / std_val
    
    def align_signals_cross_correlation(self, ecg: np.ndarray, ppg: np.ndarray) -> Tuple[np.ndarray, int]:
        correlation = signal.correlate(ecg, ppg, mode="full")
        lags = signal.correlation_lags(len(ecg), len(ppg), mode="full")
        optimal_lag = lags[np.argmax(correlation)]
        aligned_ppg = np.roll(ppg, shift=optimal_lag)
        return aligned_ppg, optimal_lag 

    def preprocessing_PPG(self, ppg_signal: np.ndarray) -> np.ndarray:
        time = np.arange(len(ppg_signal)) / self.fs
        ppg_ac = self._dc_removal(time, ppg_signal)
        ppg_normalized = self.normalize_signal(ppg_ac)
        return ppg_normalized

    def preprocessing_ECG(self, ecg_signal: np.ndarray) -> np.ndarray:
        filtered_bandpass = self._butter_bandpass(ecg_signal, lowcut=0.5, highcut=100.0, order=5)
        filtered_notch = self._notch_filter_ecg(filtered_bandpass, notch_freq=50, Q=30)
        ecg_normalized = self.normalize_signal(filtered_notch)
        return ecg_normalized

# --- (DATA VISUALIZER CLASS) ---

class DataVisualizer:

    def __init__(self, fs: float, window_seconds: int, pause_time: float, step_size: int):
        self.fs = fs
        self.window_seconds = window_seconds
        self.pause_time = pause_time
        self.step_size = step_size
        self.window_samples = int(fs * window_seconds)

    def visualize_specific_segment(self, record_name: str, ppg_segment: np.ndarray, ecg_segment: np.ndarray):
      
        if len(ppg_segment) != len(ecg_segment):
            print(f"Error: Length of PPG ({len(ppg_segment)}) and ECG ({len(ecg_segment)}) do not match!")
            return
        
        ppg_clean = np.nan_to_num(ppg_segment)
        ecg_clean = np.nan_to_num(ecg_segment)
        
        num_samples = len(ppg_clean)
        duration = num_samples / self.fs
        x_axis = np.linspace(0, duration, num_samples)

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        ax.plot(x_axis, ppg_clean, color="blue", linewidth=1.5, label="PPG Segment", alpha=0.8)
        ax.plot(x_axis, ecg_clean, color="red", linewidth=1.5, label="ECG Segment", alpha=0.8)

        current_min = min(np.min(ppg_clean), np.min(ecg_clean))
        current_max = max(np.max(ppg_clean), np.max(ecg_clean))
        
        margin = (current_max - current_min) * 0.1 if (current_max != current_min) else 1.0
        ax.set_ylim(current_min - margin, current_max + margin)

        ax.set_title(f"Record: {record_name} | Length: {duration:.2f}s ({num_samples} samples)")
        ax.set_ylabel("Amplitude")
        ax.set_xlabel("Time (seconds)")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.show()

    def visualize_sliding_record(self, record_name: str, ppg_signal: np.ndarray, ecg_signal: np.ndarray):
        total_samples = len(ppg_signal)
        
        if total_samples < self.window_samples:
            print(f"Record {record_name} quá ngắn ({total_samples} samples). Bỏ qua.")
            return

        plt.ion()
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        ax.set_title(f"Record: {record_name} (FS={self.fs}Hz)")
        x_axis = np.linspace(0, self.window_seconds, self.window_samples)

        line_ppg, = ax.plot(x_axis, np.zeros(self.window_samples), color="blue", linewidth=1.5, label="PPG", alpha=0.7)
        line_ecg, = ax.plot(x_axis, np.zeros(self.window_samples), color="red", linewidth=1.5, label="ECG", alpha=0.7)

        ax.set_ylabel("Normalized Amplitude")
        ax.set_xlabel("Time in Window (seconds)")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.6)


        try:
            # 3. (Sliding Window)
            for start_idx in range(0, total_samples - self.window_samples, self.step_size):
                if not plt.fignum_exists(fig.number):
                    break
                
                end_idx = start_idx + self.window_samples
                
                window_ppg = ppg_signal[start_idx:end_idx]
                window_ecg = ecg_signal[start_idx:end_idx]
                
                window_ppg = np.nan_to_num(window_ppg)
                window_ecg = np.nan_to_num(window_ecg)

                line_ppg.set_ydata(window_ppg)
                line_ecg.set_ydata(window_ecg)
                
                current_min = min(np.min(window_ppg), np.min(window_ecg))
                current_max = max(np.max(window_ppg), np.max(window_ecg))
                
                margin = (current_max - current_min) * 0.1 if (current_max != current_min) else 1.0
                
                ax.set_ylim(current_min - margin, current_max + margin)

                curr_time_start = start_idx / self.fs
                curr_time_end = end_idx / self.fs
                ax.set_title(f"Record: {record_name} | Time: {curr_time_start:.2f}s - {curr_time_end:.2f}s")

                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                
                if self.pause_time > 0:
                    time.sleep(self.pause_time)
                    
        except KeyboardInterrupt:
            print("\nStopped by user (KeyboardInterrupt).")
        except Exception as e:
            print(f"Error occurred: {e}")
        finally:
            plt.close(fig)
            plt.ioff() 
            print("Visualization complete.")