import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from load_data import LoadData
from CLIP import ECGDecoder_UNet, ECGEssembleCLIP
import random
import os
from metric import calculate_cosine_similarity, calculate_dtw_distance, calculate_metrics


# --- CONFIGURATION ---
SEED = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16 
INPUT_LENGTH = 2400
OUTPUT_EMBED_DIM = 128
TEST_DATA_PATH = 'processed_data/mimic3_v1_test.npz'
CLIP_MODEL_PATH = "multitask_clip_best_model_mimiciii.pth"
DECODER_MODEL_PATH = "multitask_decoder_best_model_mimiciii.pth"
ppg_sqi_thresh = 0.3
ecg_sqi_thresh = 0.3
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_models():
    print(f"Loading models on {DEVICE}...")
    
    model_clip = ECGEssembleCLIP(embed_dim=OUTPUT_EMBED_DIM).to(DEVICE)
    model_converter = ECGDecoder_UNet(bottleneck_channels=2048, target_length=2400).to(DEVICE)

    if torch.cuda.is_available():
        clip_weights = torch.load(CLIP_MODEL_PATH)
        decoder_weights = torch.load(DECODER_MODEL_PATH)
    else:
        clip_weights = torch.load(CLIP_MODEL_PATH, map_location='cpu')
        decoder_weights = torch.load(DECODER_MODEL_PATH, map_location='cpu')

    model_clip.load_state_dict(clip_weights)
    model_converter.load_state_dict(decoder_weights)
    
    model_clip.eval()
    model_converter.eval()
    
    return  model_clip, model_converter

def save_ecg_reconstruction(output_path = "AF_Detection/ecg_reconstructions.npz"):
    set_seed(SEED)
    model_clip, model_converter = load_models()
    all_predicted_ecgs = []
    all_original_ppgs = []
    all_labels = []
    all_record_names = []

    try:
        test_dataset = LoadData(TEST_DATA_PATH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    except Exception as e:
        print(f"Error: {e}")
        return
   
    print("Running inference for visualization...")
    with torch.no_grad():
        for i, (ecg, ppg, record_names, label) in enumerate(test_loader):
            ppg_input = ppg.to(DEVICE).float().unsqueeze(1)
            print("record name: ", record_names)
            ppg_embedding, feature_lists_PPG = model_clip(None, ppg_input)
            predicted_ecg = model_converter(ppg_embedding, feature_lists_PPG)
            all_predicted_ecgs.append(predicted_ecg.squeeze(1).cpu().numpy())
            all_original_ppgs.append(ppg.cpu().numpy())
            all_labels.append(label.cpu().numpy())
            all_record_names.append(record_names)
    save_dict = {
        "ecgs": np.concatenate(all_predicted_ecgs, axis=0),
        "ppgs": np.concatenate(all_original_ppgs, axis=0),
        "labels": np.concatenate(all_labels, axis=0),
        "records": np.array(all_record_names, dtype=object) 
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, **save_dict)


def visualize_results(ppg, ecg_true, ecg_pred, sample_idx, record_name):
    print(f"Visualizing Record: {record_name}")
    t_ppg = np.arange(len(ppg))
    t_ecg = np.arange(len(ecg_true))

    plt.figure(figsize=(12, 10))
    
    plt.suptitle(f"Record: {record_name} - Sample ID: {sample_idx}", fontsize=14, fontweight='bold')

    plt.subplot(4, 1, 1)
    plt.plot(t_ppg, ppg, color='green', label='Input PPG')
    plt.title("Input PPG Signal")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    plt.subplot(4, 1, 2)
    plt.plot(t_ecg, ecg_true, color='blue', label='Ground Truth ECG')
    plt.title("Ground Truth ECG")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    plt.subplot(4, 1, 3)
    plt.plot(t_ecg, ecg_pred, color='red', label='Predicted ECG')
    plt.title("Predicted ECG (Reconstructed)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    plt.subplot(4, 1, 4)
    plt.plot(t_ecg, ecg_true, color='black', label='Ground Truth', alpha=0.7)
    plt.plot(t_ecg, ecg_pred, color='red', label='Predicted', linestyle='--', alpha=0.8)
    plt.title("Comparison: Ground Truth vs Predicted ECG")
    plt.xlabel("Time Samples")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def run_visualization():
    set_seed(SEED)
    model_clip, model_converter = load_models()
    seen_records = set()

    try:
        test_dataset = LoadData(TEST_DATA_PATH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    except Exception as e:
        print(f"Error: {e}")
        return
   
    print("Running inference for visualization...")
    with torch.no_grad():
        for i, (ecg, ppg, record_names) in enumerate(test_loader):
            ppg_input = ppg.to(DEVICE).float().unsqueeze(1)
            
            ppg_embedding, feature_lists_PPG = model_clip(None, ppg_input)
            predicted_ecg = model_converter(ppg_embedding, feature_lists_PPG)

            # np.atleast_2d vẫn được giữ lại để đảm bảo không lỗi khi batch size = 1
            ppg_np = np.atleast_2d(ppg_input.cpu().squeeze().numpy())
            ecg_true_np = np.atleast_2d(ecg.cpu().squeeze().numpy())
            ecg_pred_np = np.atleast_2d(predicted_ecg.cpu().squeeze().numpy())

            for idx in range(ppg_np.shape[0]):
                current_rec_name = record_names[idx]
                
                if current_rec_name not in seen_records:
                    visualize_results(
                        ppg_np[idx], 
                        ecg_true_np[idx], 
                        ecg_pred_np[idx], 
                        len(seen_records), 
                        current_rec_name
                    )
                    seen_records.add(current_rec_name)

def run_loss():
    model_clip, model_converter = load_models()
    test_dataset = LoadData(TEST_DATA_PATH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    total_rmse, total_pearson, total_dtw, total_cosine, total_samples = 0, 0, 0, 0, 0

    print("Calculating Metrics...")
    with torch.no_grad():
        for i, (ecg, ppg, record_names) in enumerate(test_loader):
            ppg_input = ppg.to(DEVICE).float().unsqueeze(1)
            
            ppg_embedding, feature_lists_PPG = model_clip(None, ppg_input)
            predicted_ecg = model_converter(ppg_embedding, feature_lists_PPG)

            ecg_true_np = np.atleast_2d(ecg.cpu().squeeze().numpy())
            ecg_pred_np = np.atleast_2d(predicted_ecg.cpu().squeeze().numpy())

            for b in range(ecg_true_np.shape[0]):
                true_s = ecg_true_np[b]
                pred_s = ecg_pred_np[b]

                rmse, pearson = calculate_metrics(true_s, pred_s)
                dtw = calculate_dtw_distance(true_s, pred_s)
                cosine = calculate_cosine_similarity(true_s, pred_s)
                # print("Record:", record_names[b])
                # print(f"  rRMSE: {rmse:.4f}, Pearson: {pearson:.4f}, DTW: {dtw:.4f}, Cosine: {cosine:.4f}")
                total_rmse += rmse
                total_pearson += pearson
                total_dtw += dtw
                total_cosine += cosine
                total_samples += 1

    print(f"\nFinal Results ({total_samples} samples):")
    print(f"rRMSE: {total_rmse/total_samples:.4f}")
    print(f"Pearson: {total_pearson/total_samples:.4f}")
    print(f"DTW: {total_dtw/total_samples:.4f}")
    print(f"Cosine: {total_cosine/total_samples:.4f}")


def save_ecg_reconstruction_deepbeat(output_path = "AF_Detection/ecg_deepbeat_reconstructions.npz"):
    set_seed(SEED)
    model_clip, model_converter = load_models()
    all_predicted_ecgs = []
    all_original_ppgs = []
    all_labels = []

    try:
        test_dataset = LoadData(TEST_DATA_PATH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    except Exception as e:
        print(f"Error: {e}")
        return
   
    print("Running inference for visualization...")
    with torch.no_grad():
        for i, ( ppg, label) in enumerate(test_loader):
            ppg_input = ppg.to(DEVICE).float().unsqueeze(1)
            ppg_embedding, feature_lists_PPG = model_clip(None, ppg_input)
            predicted_ecg = model_converter(ppg_embedding, feature_lists_PPG)
            all_predicted_ecgs.append(predicted_ecg.squeeze(1).cpu().numpy())
            all_original_ppgs.append(ppg.cpu().numpy())
            all_labels.append(label.cpu().numpy())
    save_dict = {
        "ecgs": np.concatenate(all_predicted_ecgs, axis=0),
        "ppgs": np.concatenate(all_original_ppgs, axis=0),
        "labels": np.concatenate(all_labels, axis=0),
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, **save_dict)
if __name__ == "__main__":
    run_visualization()
    # run_loss()
    # save_ecg_reconstruction("AF_Detection/total_ecg_reconstructions.npz")
    # save_ecg_reconstruction_deepbeat("AF_Detection/deepbeat_ecg_reconstructions.npz")