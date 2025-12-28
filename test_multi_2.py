import torch
import torch.nn as nn # Thêm module nn để dùng hàm MSELoss
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from load_data import LoadData
from CLIP import ECGDecoder_UNet, ECGEssembleCLIP

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16 
INPUT_LENGTH = 2400
OUTPUT_EMBED_DIM = 128
TEST_DATA_PATH = 'datasets/normal_test.npz' 
CLIP_MODEL_PATH = "multitask_clip_best_model.pth"
DECODER_MODEL_PATH = "multitask_decoder_best_model.pth"
NUM_SAMPLES_TO_PLOT = 5 

def load_models():
    """Initializes models and loads the trained weights."""
    print(f"Loading models on {DEVICE}...")
    
    # 1. Initialize Architecture
    model_clip = ECGEssembleCLIP(embed_dim=OUTPUT_EMBED_DIM).to(DEVICE)
    model_converter = ECGDecoder_UNet(
        bottleneck_channels=2048
    ).to(DEVICE)

    # 2. Load Weights
    if torch.cuda.is_available():
        clip_weights = torch.load(CLIP_MODEL_PATH)
        decoder_weights = torch.load(DECODER_MODEL_PATH)
    else:
        clip_weights = torch.load(CLIP_MODEL_PATH, map_location=torch.device('cpu'))
        decoder_weights = torch.load(DECODER_MODEL_PATH, map_location=torch.device('cpu'))

    model_clip.load_state_dict(clip_weights)
    model_converter.load_state_dict(decoder_weights)
    
    model_clip.eval()
    model_converter.eval()
    
    return model_clip, model_converter

def visualize_results(ppg, ecg_true, ecg_pred, sample_idx):
    """
    Draws 4 subplots: PPG, True ECG, Predicted ECG, Error Signal.
    """
    error_signal = ecg_true - ecg_pred
    t = np.arange(len(ppg))

    plt.figure(figsize=(12, 10))
    plt.suptitle(f"Inference Result - Sample {sample_idx}", fontsize=16)

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
    plt.plot(t, ecg_pred, color='red', linestyle='--', label='Predicted ECG')
    plt.title("Predicted ECG (Reconstructed)")
    plt.ylabel("Amplitude")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    # 4. Error Signal
    plt.subplot(4, 1, 4)
    plt.plot(t, error_signal, color='purple', label='Error (True - Pred)')
    plt.fill_between(t, error_signal, color='purple', alpha=0.2) 
    plt.title("Error Signal")
    plt.xlabel("Time Samples")
    plt.ylabel("Difference")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def run_test():
    # Load Models
    model_clip, model_converter = load_models()

    # Load Data
    try:
        test_dataset = LoadData(TEST_DATA_PATH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True) 
        print(f"Loaded dataset from {TEST_DATA_PATH}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # --- BỔ SUNG: Khởi tạo hàm loss và biến đếm ---
    criterion = nn.MSELoss() 
    total_mse = 0.0
    total_samples = 0

    # Inference Loop
    print("Running inference...")
    with torch.no_grad():
        for i, (ecg, ppg, labels, groupID) in enumerate(test_loader):
            # Move to device and add channel dim
            ecg_target = ecg.to(DEVICE).float().unsqueeze(1)
            ppg_input = ppg.to(DEVICE).float().unsqueeze(1)

            # --- Forward Pass ---
            # 1. Get embedding from CLIP 
            _, ppg_embedding, feature_lists_PPG = model_clip(ecg_target, ppg_input)

            # 2. Decode embedding to ECG using Converter
            predicted_ecg = model_converter(ppg_embedding, feature_lists_PPG)

            # --- BỔ SUNG: Tính toán MSE Loss ---
            loss = criterion(predicted_ecg, ecg_target)
            
            # Cộng dồn loss (nhân với batch size hiện tại để tính trung bình chính xác sau này)
            current_batch_size = ecg_target.size(0)
            total_mse += loss.item() * current_batch_size
            total_samples += current_batch_size

            # --- Visualization ---
            # (Chỉ vẽ batch đầu tiên để không bị spam hình)
            if i == 0: 
                ppg_np = ppg_input.cpu().squeeze().numpy()
                ecg_true_np = ecg_target.cpu().squeeze().numpy()
                ecg_pred_np = predicted_ecg.cpu().squeeze().numpy()

                if len(ppg_np.shape) > 1:
                    batch_samples = ppg_np.shape[0]
                    samples_to_show = min(batch_samples, NUM_SAMPLES_TO_PLOT)
                    
                    for idx in range(samples_to_show):
                        visualize_results(
                            ppg_np[idx], 
                            ecg_true_np[idx], 
                            ecg_pred_np[idx], 
                            idx
                        )
                else:
                    # Single item batch
                    visualize_results(ppg_np, ecg_true_np, ecg_pred_np, 0)

    # --- BỔ SUNG: Tính và in ra MSE trung bình toàn tập test ---
    avg_mse = total_mse / total_samples
    print("="*40)
    print(f"TEST COMPLETED")
    print(f"Total Samples evaluated: {total_samples}")
    print(f"Average MSE Loss on Test Set: {avg_mse:.6f}")
    print("="*40)

if __name__ == "__main__":
    run_test()