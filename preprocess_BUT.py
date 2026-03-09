import os
import numpy as np
import wfdb
import csv
import matplotlib.pyplot as plt
from preprocessing import SignalProcessor
from utils import calculate_bsqi, calculate_sq_mask

signal_preprocess_ppg = SignalProcessor(fs=30)
signal_preprocess_ecg = SignalProcessor(fs=1000)

def mask_filter(ecg, ppg): 
    sq_mask_array = calculate_sq_mask(ppg, fs=30)
    ppg_sqi = np.mean(sq_mask_array)
    ecg_sqi = calculate_bsqi(ecg, fs=1000)
    if ppg_sqi < 0.3 or ecg_sqi < 0.3:
        return True
    return False

def preprocess_PPG(ppg_signal):
    time = np.arange(len(ppg_signal)) / 30
    filtered_ppg = signal_preprocess_ppg._butter_bandpass(ppg_signal, lowcut=0.5, highcut=8.0, order=5)
    dc_remov_ppg = signal_preprocess_ppg._dc_removal(time, filtered_ppg)
    normalized_ppg = signal_preprocess_ppg.normalize_signal(dc_remov_ppg)
    return normalized_ppg

def preprocess_ECG(ecg_signal):
    filtered_ecg = signal_preprocess_ecg._butter_bandpass(ecg_signal, lowcut=0.5, highcut=100.0, order=5)
    filtered_notch = signal_preprocess_ecg._notch_filter_ecg(filtered_ecg, notch_freq=50, Q=30)
    normalized_ecg = signal_preprocess_ecg.normalize_signal(filtered_notch)
    return normalized_ecg

def load_signals(datapath):
    all_data = []

    if os.path.isdir(datapath):
        record_files = [f for f in os.listdir(datapath) if f.endswith('.dat')]
    else:
        print(f"Warning: {datapath} is not a directory.")
        return 

    base_ids = set([f.split('_')[0] for f in record_files])

    for base_id in base_ids:
        ecg_path = os.path.join(datapath, f"{base_id}_ECG")
        ppg_path = os.path.join(datapath, f"{base_id}_PPG")

        if not (os.path.exists(ecg_path + ".dat") and os.path.exists(ppg_path + ".dat")):
            print(f"⚠️ Missing pair for {base_id}, skipping...")
            continue

        try:
            ecg_record = wfdb.rdrecord(ecg_path)
            ppg_record = wfdb.rdrecord(ppg_path)

        except Exception as e:
            print(f"❌ Error reading {base_id}: {e}")
            continue
        # print("ppg_record: ", ppg_record)
        # print(ppg_record.__dict__.keys())
        # for key, value in vars(ppg_record).items():
        #     print(f"{key}: {value}")
        
        ecg_data = ecg_record.p_signal
        ppg_data = ppg_record.p_signal
        if ecg_data.shape[0] == 1:
            ecg_signal = ecg_data[0]
        else:
            ecg_signal = ecg_data[:, 0]

        if ppg_data.shape[0] == 1:
            ppg_signal = ppg_data[0]
        elif ppg_data.shape[1] == 3:
            ppg_signal = ppg_data[:, 1] 
        else:
            ppg_signal = ppg_data[:, 0]

        fs_ecg = ecg_record.fs   #  1000 Hz
        fs_ppg = ppg_record.fs   #  30 Hz
        # print("shape of signal ECG and PPG: ",  ecg_record.p_signal.shape, ppg_record.p_signal.shape)
        # print(f"Loaded {base_id} - ECG length: {len(ecg_signal)}, PPG length: {len(ppg_signal)}, fs_ecg: {fs_ecg}, fs_ppg: {fs_ppg}")
        if len(ecg_signal) <=1 or len(ppg_signal) <=1:
            continue
        ecg_signal = preprocess_ECG(ecg_signal)
        ppg_signal = preprocess_PPG(ppg_signal)
        if mask_filter(ecg_signal, ppg_signal):
            continue
        all_data.append({
                'id': base_id,
                'ecg': ecg_signal,
                'ppg': ppg_signal,
                'fs_ecg': ecg_record.fs,
                'fs_ppg': ppg_record.fs
            })
        # print("base_id: ", base_id)
    return all_data

def visualize_signals(data_dict, duration_sec=10):
   
    ecg = data_dict['ecg']
    ppg = data_dict['ppg']
    fs_e = data_dict['fs_ecg']
    fs_p = data_dict['fs_ppg']
    
    n_samples_ecg = int(duration_sec * fs_e)
    n_samples_ppg = int(duration_sec * fs_p)
    
    ecg_plot = ecg[:n_samples_ecg]
    ppg_plot = ppg[:n_samples_ppg]
    
    t_ecg = np.arange(n_samples_ecg) / fs_e
    t_ppg = np.arange(n_samples_ppg) / fs_p
    
    plt.figure(figsize=(15, 7))
    
    plt.plot(t_ppg, ppg_plot, color='blue', label='PPG Segment', linewidth=1.2, alpha=0.8)
    plt.plot(t_ecg, ecg_plot, color='#d63031', label='ECG Segment', linewidth=1.2, alpha=0.9)
    
    plt.title(f"Record: {data_dict.get('id', 'mimic_perform')} | Length: {duration_sec:.2f}s ({n_samples_ecg} samples)", fontsize=12)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def split_and_save(total_data, save_prefix="but"):
    num_samples = len(total_data["ppgs"])
    
    indices = np.random.permutation(num_samples)
    
    split_idx = int(num_samples * 0.8)
    
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    os.makedirs("processed_data", exist_ok=True)
    
    total_samples = len(total_data["ppgs"])
    print(f"\nTotal: {total_samples})")
    print("-" * 50)

    for name, idxs in [("train", train_idx), ("test", test_idx)]:
        save_dict = {
            "ppgs": [total_data["ppgs"][i] for i in idxs],
            "ecgs": [total_data["ecgs"][i] for i in idxs],
            "records": [total_data["records"][i] for i in idxs]
        }
        
        num_samples = len(idxs)
        percentage = (num_samples / total_samples) * 100
        unique_recs = len(set(save_dict["records"])) 
        
        save_path = f"processed_data/{save_prefix}_{name}.npz"
        np.savez(save_path, **save_dict)
        
        print(f" {name.upper()}:")
        print(f"   - Number samples: {num_samples} ({percentage:.1f}%)")
        print(f"   - Unique records: {unique_recs}")
        print(f"   - Saved at: {save_path}")
        print("-" * 50)

    print("Finished splitting and summarizing data!")

def visualize_3_channel_ppg(record_id, base_path):
    ppg_path = os.path.join(base_path, str(record_id), f"{record_id}_PPG")
    
    try:
        record = wfdb.rdrecord(ppg_path)
        signal = record.p_signal  # (300, 3)
        fs = record.fs            # 30Hz
        
        time = np.arange(signal.shape[0]) / fs
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        colors = ['#000000', '#000000', '#000000']  # Neon Pink, Cyan, Electric Purple
        labels = ['Channel 1', 'Channel 2', 'Channel 3']

        for i in range(3):
            signal_ppg = preprocess_PPG(signal[:, i])  
            axes[i].plot(time, signal_ppg, color=colors[i], label=labels[i])
            axes[i].set_ylabel("Amplitude")
            axes[i].set_title(f"PPG {record_id} - {labels[i]}")
            axes[i].legend(loc='upper right')
            axes[i].grid(True, linestyle='--', alpha=0.6)

        axes[2].set_xlabel("Time (seconds)")
        plt.tight_layout()
        plt.show()
        
        print(f"Successfully loaded record {record_id}")
        print(f"Signal shape: {signal.shape}, Sampling Rate: {fs}Hz")

    except Exception as e:
        print(f"Error loading file {record_id}: {e}")

def plot_single_channel_ppg(record_id, base_path):
    ppg_path = os.path.join(base_path, str(record_id), f"{record_id}_PPG")
    
    try:
        record = wfdb.rdrecord(ppg_path)
        signal = record.p_signal.flatten() 
        signal_ppg = preprocess_PPG(signal)
        fs = record.fs 
        
        time = np.arange(len(signal)) / fs
        
        plt.figure(figsize=(12, 4))
        plt.plot(time, signal_ppg, color='#2ecc71', linewidth=1.5)
        
        plt.title(f"PPG Signal - Record {record_id} (1 Channel)")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.xlim(0, 10) 
        
        plt.tight_layout()
        plt.show()
        
        print(f" Loaded {record_id}: {len(signal)} samples at {fs}Hz ({len(signal)/fs}s)")

    except Exception as e:
        print(f" Error: {e}")

if __name__ == "__main__":
    all_ppgs, all_ecgs, all_records = [], [], []
    dataset = "/home/linhhima/Pre_processing_data/brno-university-of-technology-smartphone-ppg-database-but-ppg-2.0.0/"
    annotation_file = "/home/linhhima/Pre_processing_data/brno-university-of-technology-smartphone-ppg-database-but-ppg-2.0.0/quality-hr-ann.csv"
    ppgs = []
    ecgs = []
    filelist = []
    with open(annotation_file, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader) 
        
        for row in reader:
            if len(row) >= 2:
                key = row[0]   
                value = row[1] 
                if value == "1":
                    filelist.append(key)
   
    print("total number of files: ", len(filelist))
    for file in filelist:
        name_file = os.path.join(dataset, file)
        print("name_file: ", file)
        signals = load_signals(name_file)
        # if signals:
        #     visualize_signals(signals[0], duration_sec=10)
        all_ppgs.extend([s['ppg'] for s in signals])
        all_ecgs.extend([s['ecg'] for s in signals])
        all_records.extend([s['id'] for s in signals])
    total_data = {
        "ppgs": all_ppgs,
        "ecgs": all_ecgs,  
        "records": all_records        
    }
    # split_and_save(total_data, save_prefix="but")
    # DATASET_DIR = "/home/linhhima/Pre_processing_data/brno-university-of-technology-smartphone-ppg-database-but-ppg-2.0.0/"
    # RECORD_ID = "138032"

    # visualize_3_channel_ppg(RECORD_ID, DATASET_DIR)

    # DATASET_ROOT = "/home/linhhima/Pre_processing_data/brno-university-of-technology-smartphone-ppg-database-but-ppg-2.0.0/"
    # RECORD_ID = "100001"

    # plot_single_channel_ppg(RECORD_ID, DATASET_ROOT)