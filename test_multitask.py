import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from load_data import LoadData
from CLIP import ECGDecoder_UNet, ECGEssembleCLIP, PPGtoECGConverter
import random
import os

# --- CONFIGURATION ---
SEED = 44
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16 # Small batch for testing
INPUT_LENGTH = 2400
OUTPUT_EMBED_DIM = 128
TEST_DATA_PATH = 'datasets/normal_test.npz' # Change to normal_train.npz if val doesn't exist yet
CLIP_MODEL_PATH = "multitask_clip_best_model.pth"
DECODER_MODEL_PATH = "multitask_decoder_best_model.pth"
NUM_SAMPLES_TO_PLOT = 4 

def set_seed(seed_value: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    print(f"Random seed set to: {seed_value}")

def load_models():
    """Initializes models and loads the trained weights."""
    print(f"Loading models on {DEVICE}...")
    
    # 1. Initialize Architecture
    model_clip = ECGEssembleCLIP(embed_dim=OUTPUT_EMBED_DIM).to(DEVICE)
    model_converter = ECGDecoder_UNet(
        bottleneck_channels=2048
    ).to(DEVICE)

    # 2. Load Weights
    # Load CLIP (Encoder)
    if torch.cuda.is_available():
        clip_weights = torch.load(CLIP_MODEL_PATH)
        decoder_weights = torch.load(DECODER_MODEL_PATH)
    else:
        clip_weights = torch.load(CLIP_MODEL_PATH, map_location=torch.device('cpu'))
        decoder_weights = torch.load(DECODER_MODEL_PATH, map_location=torch.device('cpu'))

    model_clip.load_state_dict(clip_weights)
    
    # CRITICAL: In train.py, you saved only model_converter.ecg_decoder
    model_converter.load_state_dict(decoder_weights)
    
    model_clip.eval()
    model_converter.eval()
    
    return model_clip, model_converter



def visualize_results(ppg, ecg_true, ecg_pred, sample_idx, record_name):
    """
    Draws 4 subplots: PPG, True ECG, Predicted ECG, Error Signal.
    Includes record_name in the title.
    """
    # Calculate Error Signal (Difference)
    error_signal = ecg_true - ecg_pred
    
    # Create Time Axis (optional, assuming indices)
    t = np.arange(len(ppg))

    plt.figure(figsize=(12, 10))
    
    # --- CẬP NHẬT: Thêm tên Record vào tiêu đề chính ---
    plt.suptitle(f"Record: {record_name}", fontsize=16, fontweight='bold')

    # 1. PPG Input
    plt.subplot(4, 1, 1)
    plt.plot(t, ppg, color='green', label='Input PPG')
    plt.title("Input PPG Signal")
    plt.ylabel("Amplitude")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    # 2. Ground Truth ECG
    plt.subplot(4, 1, 2)
    plt.plot(t, ecg_true, color='blue', label='Ground Truth ECG')
    plt.title("Ground Truth ECG")
    plt.ylabel("Amplitude")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    # 3. Predicted ECG
    plt.subplot(4, 1, 3)
    plt.plot(t, ecg_pred, color='red', label='Predicted ECG')
    plt.title("Predicted ECG (Reconstructed)")
    plt.ylabel("Amplitude")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    # 4. Error Signal
    plt.subplot(4, 1, 4)
    plt.plot(t, error_signal, color='purple', label='Error (True - Pred)')
    plt.fill_between(t, error_signal, color='purple', alpha=0.2) # Shading
    plt.title("Error Signal")
    plt.xlabel("Time Samples")
    plt.ylabel("Difference")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def run_test():
    set_seed(SEED)
    # Load Models
    model_clip, model_converter = load_models()

    # Load Data
    try:
        test_dataset = LoadData(TEST_DATA_PATH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True) # Shuffle to get random samples
        print(f"Loaded dataset from {TEST_DATA_PATH}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return


    seen_records = set()
    samples_collected = 0
    TARGET_SAMPLES = 4
    print(f"Running inference to find {TARGET_SAMPLES} unique records...")
    # Inference Loop
    print("Running inference...")
    with torch.no_grad():
        for i, (ecg, ppg, labels, groupID, record_names) in enumerate(test_loader):
            # Move to device and add channel dim
            ecg_target = ecg.to(DEVICE).float().unsqueeze(1)
            ppg_input = ppg.to(DEVICE).float().unsqueeze(1)

            # --- Forward Pass ---
            # 1. Get embedding from CLIP (we ignore logits here)
            _, ppg_embedding, feature_lists_PPG = model_clip(ecg_target, ppg_input)

            # 2. Decode embedding to ECG using Converter
            predicted_ecg = model_converter(ppg_embedding, feature_lists_PPG)

            # --- Visualization ---
            # Convert to CPU numpy for plotting
            ppg_np = ppg_input.cpu().squeeze().numpy()
            ecg_true_np = ecg_target.cpu().squeeze().numpy()
            ecg_pred_np = predicted_ecg.cpu().squeeze().numpy()

            current_batch_size = ppg_np.shape[0]
            for idx in range(current_batch_size):
                # Lấy tên record tương ứng
                current_rec_name = record_names[idx]

                # Kiểm tra xem record này đã vẽ chưa
                if current_rec_name not in seen_records:
                    print(f"Found unique record [{samples_collected + 1}/{TARGET_SAMPLES}]: {current_rec_name}")
                    
                    # Gọi hàm visualize
                    visualize_results(
                        ppg_np[idx], 
                        ecg_true_np[idx], 
                        ecg_pred_np[idx], 
                        samples_collected,  # Dùng biến đếm làm ID cho hình ảnh
                         current_rec_name
                    )
                    
                    # Đánh dấu đã xem và tăng biến đếm
                    seen_records.add(current_rec_name)
                    samples_collected += 1

                    # Điều kiện dừng: Đã đủ 4 mẫu khác nhau
                    if samples_collected >= TARGET_SAMPLES:
                        print("Done plotting 4 unique records.")
                        return

if __name__ == "__main__":
    
    run_test()