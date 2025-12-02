import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from load_data import LoadData # Assuming this is a custom class you have
import os

# --- 1. Configuration ---
WINDOW_SIZE = 2400      # How many time steps to look at in one go
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 50
BEST_MODEL_PATH = 'best_model.pth' # Path to save the best model

# Define paths
DATA_DIR = 'datasets/'
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'normal_train.npz')
VAL_DATA_PATH = os.path.join(DATA_DIR, 'normal_train.npz')
TEST_DATA_PATH = os.path.join(DATA_DIR, 'normal_test.npz') # <-- MAKE SURE THIS FILE EXISTS

# --- 2. The MLP Model Definition ---
class PpgToEcgMLP(nn.Module):
    def __init__(self, input_size):
        super(PpgToEcgMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, input_size) # Output size must match input window size
        )
    
    def forward(self, x):
        return self.layers(x)

# --- 3. Data Loading ---
try:
    train_dataset = LoadData(TRAIN_DATA_PATH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    val_dataset = LoadData(VAL_DATA_PATH)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # --- NEW: Load the Test data ---
    test_dataset = LoadData(TEST_DATA_PATH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Data ready: {len(train_dataset)} train samples, {len(val_dataset)} val samples.")
    print(f"Test samples: {len(test_dataset)}")

except Exception as e:
    print(f"Error loading data: {e}")
    print("Please ensure 'load_data.py' is in the same directory and all file paths are correct.")
    exit() # Exit if data can't be loaded

# --- 4. Model Initialization and Training ---

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = PpgToEcgMLP(input_size=WINDOW_SIZE).to(device)
criterion = nn.MSELoss()  # Mean Absolute Error for signal regression
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# best_val_loss = float('inf') 
best_train_loss = float('inf')
# Lists to store loss history for plotting
train_loss_history = []
# val_loss_history = []

print("Starting training...")
for epoch in range(EPOCHS):
    # --- Training Phase ---
    model.train()
    train_loss = 0
    for ecg_batch, ppg_batch, _, _ in train_loader: 
        ppg_batch, ecg_batch = ppg_batch.to(device), ecg_batch.to(device)
        
        outputs = model(ppg_batch)
        loss = criterion(outputs, ecg_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    train_loss_history.append(avg_train_loss)
    
    # # --- Validation Phase ---
    # # model.eval()
    # model.train()

    # val_loss = 0
    # with torch.no_grad():
    #     for ecg_batch,ppg_batch, _ in val_loader:
    #         ppg_batch, ecg_batch = ppg_batch.to(device), ecg_batch.to(device)
    #         outputs = model(ppg_batch)
    #         loss = criterion(outputs, ecg_batch)
    #         val_loss += loss.item()
    
    # avg_val_loss = val_loss / len(val_loader)
    # val_loss_history.append(avg_val_loss)
    
    # print(f'Epoch [{epoch+1:02d}/{EPOCHS}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
    print(f'Epoch [{epoch+1:02d}/{EPOCHS}], Train Loss: {avg_train_loss:.6f}')
    # --- Save the best model (Dựa trên Training Loss) ---
    if avg_train_loss < best_train_loss:
        best_train_loss = avg_train_loss
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f'  -> New best model saved (Best Train Loss: {best_train_loss:.6f})')
    # # --- Save the best model ---
    # if avg_val_loss < best_val_loss:
    #     best_val_loss = avg_val_loss
    #     torch.save(model.state_dict(), BEST_MODEL_PATH)
    #     print(f'  -> New best model saved with val_loss: {best_val_loss:.6f}')

print("Training finished.")

# --- 5. Visualize Training History ---
plt.figure(figsize=(12, 5))
plt.plot(train_loss_history, label='Training Loss')
# plt.plot(val_loss_history, label='Validation Loss')
plt.title('Model Training History')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error (MSE) Loss')
plt.legend()
plt.grid(True)
plt.show()

# # --- 6. NEW: Final Test Evaluation (Calculating Test Loss) ---

# print(f"Loading best model (val_loss: {best_val_loss:.6f}) from {BEST_MODEL_PATH} for final testing...")
model.load_state_dict(torch.load(BEST_MODEL_PATH))
model.to(device)
model.eval() # Put model in evaluation mode

test_loss = 0
with torch.no_grad(): # No gradients needed for testing
    for ecg_batch, ppg_batch, _,_ in test_loader:
        ppg_batch, ecg_batch = ppg_batch.to(device), ecg_batch.to(device)
        
        outputs = model(ppg_batch)
        loss = criterion(outputs, ecg_batch)
        test_loss += loss.item()

avg_test_loss = test_loss / len(test_loader)

print("--------------------------------------------------")
print(f"           FINAL MODEL PERFORMANCE")
# print(f"     Best Validation Loss (MSE): {best_val_loss:.6f}")
print(f"         Average Validation Loss (MSE): {avg_test_loss:.6f}")
print("--------------------------------------------------")


# --- 7. Evaluation and Conversion Error Visualization (on one Test batch) ---

print("Running visualization on one batch from the TEST set...")
try:
    with torch.no_grad():
        # Get one batch from the test set
        ecg_actual, ppg_input, _,_ = next(iter(test_loader)) 
        ppg_input = ppg_input.to(device)
        
        # Get the model's prediction
        ecg_predicted = model(ppg_input).cpu()
        
        # --- Get data for the FIRST sample in the batch for plotting ---
        ppg_plot = ppg_input[0].cpu().numpy()
        ecg_actual_plot = ecg_actual[0].cpu().numpy()
        ecg_predicted_plot = ecg_predicted[0].numpy()
        
        # --- Calculate the error signal ---
        error_signal = ecg_actual_plot - ecg_predicted_plot
        sample_mae = np.mean(np.abs(error_signal))
        mse = np.mean((error_signal) ** 2)
        rmse = np.sqrt(mse)
        print(f"Sample MSE: {mse:.4f}, RMSE: {rmse:.4f}")
        # --- Create a 2-panel plot ---
        fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        # --- Top Plot: The Conversion ---
        axes[0].set_title(f"PPG-to-ECG Conversion (Test Sample 0) - Sample MSE: {mse:.4f}")
        axes[0].plot(ppg_plot, label="Input PPG Signal (Scaled)", color='blue', alpha=0.7)
        axes[0].plot(ecg_actual_plot, label="Actual ECG Target (Scaled)", color='green', linewidth=2, linestyle='--')
        axes[0].plot(ecg_predicted_plot, label="Predicted ECG (MLP)", color='red', linewidth=2)
        axes[0].legend()
        axes[0].grid(True)
        axes[0].set_ylabel("Normalized Amplitude")
        
        # --- Bottom Plot: The Error Signal ---
        axes[1].set_title("Conversion Error (Actual - Predicted)")
        axes[1].plot(error_signal, label='Error Signal', color='purple')
        axes[1].axhline(0, color='black', linestyle='--', linewidth=1) # Add a zero-line
        axes[1].legend()
        axes[1].grid(True)
        axes[1].set_ylabel("Error Amplitude")
        axes[1].set_xlabel("Time Samples (in window)")
        
        plt.tight_layout() # Adjusts plots to prevent overlap
        plt.show()

except StopIteration:
    print("\nVisualization failed: Could not get a batch from 'test_loader'. Is it empty?")