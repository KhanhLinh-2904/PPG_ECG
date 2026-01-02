import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from ecg_reconstruction import SignalReconstructor
from load_data import LoadData
from CLIP import ECGDecoder_DWT, ECGEssembleCLIP # Dùng lại Decoder DWT cũ
from feature_domain import SignalPreprocessor
import random
import os

# --- CONFIGURATION ---
SEED = 44
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16 
INPUT_LENGTH = 2400
OUTPUT_EMBED_DIM = 128
TEST_DATA_PATH = 'datasets/normal_test.npz' 

CLIP_MODEL_PATH = "multitask_clip_best_model.pth"
DECODER_MODEL_PATH = "multitask_decoder_best_model.pth"

NUM_SAMPLES_TO_PLOT = 6

def set_seed(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    print(f"Random seed set to: {seed_value}")

def load_models():
    print(f"Loading models on {DEVICE}...")
    
    # 1. Preprocessor & Detect Channels
    preprocessor = SignalPreprocessor(signal_length=INPUT_LENGTH)
    ppg_channels = preprocessor.scat_out_channels
    print(f"Detected PPG input channels: {ppg_channels}")

    # 2. Init Architecture
    model_clip = ECGEssembleCLIP(
        embed_dim=OUTPUT_EMBED_DIM,
        ppg_input_channels=ppg_channels
    ).to(DEVICE)
    
    # Dùng lại ECGDecoder_DWT (Output ra features)
    model_converter = ECGDecoder_DWT(
            bottleneck_channels=2048,
            output_channels=4,
            target_length=306
        ).to(DEVICE)
    
    # 3. Load Weights
    print(f"Loading weights from: \n - {CLIP_MODEL_PATH} \n - {DECODER_MODEL_PATH}")
    map_loc = DEVICE if torch.cuda.is_available() else torch.device('cpu')
    
    try:
        clip_weights = torch.load(CLIP_MODEL_PATH, map_location=map_loc)
        decoder_weights = torch.load(DECODER_MODEL_PATH, map_location=map_loc)
        
        model_clip.load_state_dict(clip_weights)
        model_converter.load_state_dict(decoder_weights)
    except FileNotFoundError as e:
        print(f"LỖI: Không tìm thấy file weights ({e}).")
        raise e

    model_clip.eval()
    model_converter.eval()
    
    return model_clip, model_converter

def visualize_results(ppg, ecg_true, ecg_pred, sample_idx, record_name):
    # Calculate Error
    error_signal = ecg_true - ecg_pred
    mse_val = np.mean(error_signal**2)
    
    t = np.arange(len(ppg))

    plt.figure(figsize=(12, 10))
    plt.suptitle(f"Record: {record_name} | MSE: {mse_val:.5f}", fontsize=16, fontweight='bold')

    # 1. PPG
    plt.subplot(4, 1, 1)
    plt.plot(t, ppg, color='green', label='Input PPG')
    plt.title("Input PPG Signal")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    # 2. Ground Truth
    plt.subplot(4, 1, 2)
    plt.plot(t, ecg_true, color='blue', label='Ground Truth ECG')
    plt.title("Ground Truth ECG")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    # 3. Predicted
    plt.subplot(4, 1, 3)
    plt.plot(t, ecg_pred, color='red', label='Predicted ECG')
    plt.title("Predicted ECG (Feature Domain Reconstruction)")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    # 4. Error
    plt.subplot(4, 1, 4)
    plt.plot(t, error_signal, color='purple', label='Error')
    plt.title("Error Signal")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def run_test():
    set_seed(SEED)
    
    # Khởi tạo các công cụ hỗ trợ
    reconstructor = SignalReconstructor(dwt_wavelet='db4', dwt_level=3, original_length=INPUT_LENGTH)
    preprocessor = SignalPreprocessor(signal_length=INPUT_LENGTH)
    
    # Load Models
    try:
        model_clip, model_converter = load_models()
    except Exception:
        return

    # Load Data
    try:
        test_dataset = LoadData(TEST_DATA_PATH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    seen_records = set()
    samples_collected = 0
    TARGET_SAMPLES = NUM_SAMPLES_TO_PLOT
    
    print("Running inference...")
    
    with torch.no_grad():
        for i, (ecg, ppg, labels, groupID, record_names) in enumerate(test_loader):
            ecg_target = ecg.to(DEVICE).float()
            ppg_input = ppg.to(DEVICE).float()
    
            # --- PREPROCESSING ---
            ppg_features = preprocessor.process_ppg_scattering(ppg_input)
            ecg_features = preprocessor.process_ecg_dwt(ecg_target) # Chỉ dùng để chạy CLIP encoder nếu cần
            
            ppg_features = ppg_features.to(DEVICE)
            ecg_features = ecg_features.to(DEVICE)

            # --- FORWARD PASS ---
            # 1. CLIP Encoder
            _, ppg_embedding, feature_lists_PPG = model_clip(ecg_features, ppg_features)

            # 2. Decoder -> Predict Features (Batch, 4, 306)
            predicted_ecg_features_tensor = model_converter(ppg_embedding, feature_lists_PPG)
            
            # 3. Inverse DWT -> Tái tạo tín hiệu (Batch, 2400)
            # Lưu ý: reconstructor đã được viết lại để nhận Tensor và trả về Tensor (ở các bước trước)
            # Nếu bản reconstructor của bạn trả về Tensor thì không cần .cpu().numpy() ở đây.
            # Tuy nhiên, nếu bản cũ dùng numpy, ta làm như sau:
            
            # --- Trường hợp dùng Reconstructor mới (Tensor-based) ---
            predicted_ecg = reconstructor.inverse_ecg_dwt(predicted_ecg_features_tensor)
            
            # --- VISUALIZATION ---
            ppg_np = ppg_input.cpu().numpy()
            ecg_true_np = ecg_target.cpu().numpy()
            ecg_pred_np = predicted_ecg.cpu().numpy() 

            current_batch_size = ppg_np.shape[0]
            
            for idx in range(current_batch_size):
                current_rec_name = record_names[idx]

                if current_rec_name not in seen_records:
                    print(f"Plotting record [{samples_collected + 1}/{TARGET_SAMPLES}]: {current_rec_name}")
                    
                    visualize_results(
                        ppg_np[idx], 
                        ecg_true_np[idx], 
                        ecg_pred_np[idx], 
                        samples_collected, 
                        current_rec_name
                    )
                    
                    seen_records.add(current_rec_name)
                    samples_collected += 1

                    if samples_collected >= TARGET_SAMPLES:
                        print("Done plotting unique records.")
                        return

if __name__ == "__main__":
    run_test()