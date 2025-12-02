import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import math

from CLIP import PPGtoECGConverter
from load_data import LoadData

# ===============================
# === Hyperparameters & Paths ===
# ===============================

TEST_DATA_PATH = 'datasets/normal_test.npz'
OUTPUT_EMBED_DIM = 128
INPUT_LENGTH = 2400
BATCH_SIZE = 128
MODEL_PATH = "ppg_to_ecg_converter_kullback_best_train_loss.pth"


# ========================================
# === (1) MSE TESTING FUNCTION ===========
# ========================================

def test_converter_mse(model, test_loader, device):
    """
    Evaluate the PPG→ECG model on test set using MSE loss.
    """
    model.eval()
    mse_loss = nn.MSELoss()
    total_loss = 0.0

    with torch.no_grad():
        progress = tqdm(test_loader, desc="Testing (MSE)", unit="batch")

        # for ecg, ppg, _,_ in progress:
        for ecg, ppg, _,_ in progress:

            ppg = ppg.to(device).float().unsqueeze(1)
            ecg = ecg.to(device).float().unsqueeze(1)

            with autocast():
                ecg_pred = model(ppg)
                loss = mse_loss(ecg_pred, ecg)

            total_loss += loss.item()

    torch.cuda.empty_cache()
    return total_loss / len(test_loader)


# ==========================================
# === (2) PLOT PPG→ECG FOR ONE SAMPLE ======
# ==========================================

def plot_test_sample(model, test_loader, device):
    """
    Pick one test sample, run the model, and plot:
    - Input PPG
    - Actual ECG
    - Predicted ECG
    - Error signal
    """
    print("\nPreparing test sample visualization...")
    model.eval()

    with torch.no_grad():
        try:
            # ecg_batch, ppg_batch, _,_ = next(iter(test_loader))
            ecg_batch, ppg_batch, _,_ = next(iter(test_loader))

        except StopIteration:
            print("ERROR: Test dataloader is empty!")
            return

        # Select the first sample
        ppg_in = ppg_batch[0].to(device).float().unsqueeze(0).unsqueeze(0)
        ppg_plot = ppg_batch[0].cpu().numpy()
        ecg_true = ecg_batch[0].cpu().numpy()

        # Predict ECG
        with autocast():
            ecg_pred = model(ppg_in).squeeze().cpu().numpy()

        # Calculate sample MSE error
        error_signal = ecg_true - ecg_pred
        mse_sample = np.mean(error_signal ** 2)

        # Plot
        fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

        # ---- Top plot ----
        axes[0].set_title(f"PPG→ECG Conversion (Sample 0) - MSE: {mse_sample:.4f}")
        axes[0].plot(ppg_plot, label="Input PPG", color='blue', alpha=0.7)
        axes[0].plot(ecg_true, label="Actual ECG", color='green', linestyle='--', linewidth=2)
        axes[0].plot(ecg_pred, label="Predicted ECG", color='red', linewidth=2)
        axes[0].legend()
        axes[0].grid()
        axes[0].set_ylabel("Amplitude")

        # ---- Bottom plot ----
        axes[1].set_title("Prediction Error (Actual - Predicted)")
        axes[1].plot(error_signal, label="Error", color='purple')
        axes[1].axhline(0, linestyle='--', color='black')
        axes[1].legend()
        axes[1].grid()
        axes[1].set_xlabel("Sample Index")
        axes[1].set_ylabel("Error")

        plt.tight_layout()
        plt.show()


# ======================================
# === (3) MAIN EXECUTION SCRIPT =======
# ======================================

if __name__ == '__main__':

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test set
    try:
        test_dataset = LoadData(TEST_DATA_PATH)
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        print(f"Loaded Test Dataset ({len(test_dataset)} samples).")
    except Exception as e:
        print(f"ERROR: Cannot load test dataset at {TEST_DATA_PATH}")
        print(e)
        exit()

    # Load model
    print("\nInitializing PPG→ECG model...")
    model = PPGtoECGConverter(
        embed_dim=OUTPUT_EMBED_DIM,
        input_length=INPUT_LENGTH
    ).to(device)

    print(f"Loading model weights from: {MODEL_PATH}")
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Model weights loaded successfully.")
    except Exception as e:
        print("ERROR: Cannot load model weights.")
        print(e)
        exit()

    # Run test
    mse_loss = test_converter_mse(model, test_loader, device)
    rmse_loss = math.sqrt(mse_loss)

    print("\n========== TEST RESULTS ==========")
    print(f"MSE  : {mse_loss:.8f}")
    print(f"RMSE : {rmse_loss:.8f}")
    print("==================================")

    # Plot one sample
    plot_test_sample(model, test_loader, device)
