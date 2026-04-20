import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from utils import calculate_bsqi, calculate_sq_mask
def mask_filter(ecg, ppg): 
    sq_mask_array = calculate_sq_mask(ppg, fs=125)
    ppg_sqi = np.mean(sq_mask_array)
    ecg_sqi = calculate_bsqi(ecg, fs=125)
    if ppg_sqi < 0.3 or ecg_sqi < 0.3:
        print("ppg_sqi: ", ppg_sqi)
        print("ecg_sqi: ", ecg_sqi)
        return True
    return False
def read_mat_scipy(path):
    mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    
    raw_data = mat['all_clean_data']
    
    if not isinstance(raw_data, np.ndarray):
        raw_data = np.array([raw_data])
    
    clean_list = []
    for seg in raw_data:
        d = {
            'record_name': str(seg.record_name),
            'segment_index': int(seg.segment_index),
            'fs': float(seg.fs),
            'sqi_ecg': float(seg.sqi_ecg),
            'sqi_ppg': float(seg.sqi_ppg),
            'ecg': seg.ecg,
            'ppg': seg.ppg
        }
        clean_list.append(d)
    return clean_list

def plot_signals(clean_list, num_segments=1, duration_sec=10):
  
    if not clean_list:
        print("EMPTY LIST!")
        return

    num_to_draw = min(len(clean_list), num_segments)

    for i in range(num_to_draw):
        seg = clean_list[i]
        fs = seg['fs']
        ecg = seg['ecg']
        ppg = seg['ppg']
        print(f"Record: {seg['record_name']} | Segment: {seg['segment_index']}\n")
        mask_filter(ecg, ppg)
        num_samples = int(min(len(ecg), duration_sec * fs))
        time = np.arange(num_samples) / fs

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
        
        fig.suptitle(f"Record: {seg['record_name']} | Segment: {seg['segment_index']}\n"
                     f"ECG SQI: {seg['sqi_ecg']:.2f} | PPG SQI: {seg['sqi_ppg']:.2f}", 
                     fontsize=14, fontweight='bold')

        ax1.plot(time, ecg[:num_samples], color='#1f77b4', linewidth=1)
        ax1.set_ylabel('ECG (mV)', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_title('Electrocardiogram (ECG)', loc='left', color='blue')

        ax2.plot(time, ppg[:num_samples], color='#d62728', linewidth=1)
        ax2.set_ylabel('PPG (Unit)', fontsize=12)
        ax2.set_xlabel('Time (seconds)', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.set_title('Photoplethysmogram (PPG)', loc='left', color='red')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

file_path = 'record_clean.mat'
try:
    data = read_mat_scipy(file_path)
    print(f"✅ Load {len(data)} segments.")

    plot_signals(data, num_segments=7, duration_sec=19.2)

except FileNotFoundError:
    print(f"❌ not found file: {file_path}")
except Exception as e:
    print(f"❌ error: {e}")