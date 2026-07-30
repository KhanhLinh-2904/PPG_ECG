import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import FocusedNeuralNetwork 
import os
import matplotlib.pyplot as plt
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
CHECKPOINT_DIR = "/home/linhhima/PPG_ECG/AF_Detection/checkpoints/"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def train_model(train_loader, val_loader, epochs, model, criterion, optimizer, device):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val_loss = float('inf') 
    best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    epoch_best = 0

    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train_samples = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().view(-1, 1) 
            
            outputs = model(inputs) 
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)
            total_train_samples += inputs.size(0)
            
            preds = (outputs >= 0.5).float()
            correct_train += (preds == labels).sum().item()

        epoch_train_loss = running_train_loss / total_train_samples
        epoch_train_acc = correct_train / total_train_samples

        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val_samples = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).float().view(-1, 1)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * inputs.size(0)
                total_val_samples += inputs.size(0)
                
                preds = (outputs >= 0.5).float()
                correct_val += (preds == labels).sum().item()

        epoch_val_loss = running_val_loss / total_val_samples
        epoch_val_acc = correct_val / total_val_samples

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)

        print(f"Epoch {epoch+1:03d}/{epochs} | "
              f"Train Loss: {epoch_train_loss:.6f} | Train Acc: {epoch_train_acc*100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.6f} | Val Acc: {epoch_val_acc*100:.2f}%")

        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1:03d}.pth")
        torch.save(model.state_dict(), checkpoint_path)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epoch_best = epoch + 1
            torch.save(model.state_dict(), best_model_path)

    print(f"\nComplete! Best model saved at Epoch {epoch_best} with Val Loss: {best_val_loss:.6f}.")
    return train_losses, val_losses, train_accs, val_accs

def plot_metrics(train_losses, val_losses, train_accs, val_accs):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r--', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation MSE Loss')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'r--', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_threshold_results.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    set_seed(42)
    
    train_path = '/home/linhhima/PPG_ECG/AF_Detection/detect_af_MIT_BIH_train.npz'
    val_path = '/home/linhhima/PPG_ECG/AF_Detection/detect_af_MIT_BIH_val.npz'
    
    train_data = np.load(train_path)
    val_data = np.load(val_path)
    
    X_train = torch.tensor(train_data['X'], dtype=torch.float32)
    y_train = torch.tensor(train_data['y'], dtype=torch.float32)
    
    X_val = torch.tensor(val_data['X'], dtype=torch.float32)
    y_val = torch.tensor(val_data['y'], dtype=torch.float32)

    print(f"Train Dataset: X = {X_train.shape}, y = {y_train.shape}")
    print(f"Val Dataset:   X = {X_val.shape}, y = {y_val.shape}\n")

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = FocusedNeuralNetwork()
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 200

    train_losses, val_losses, train_accs, val_accs = train_model(
        train_loader, val_loader, epochs, model, criterion, optimizer, device
    )

    plot_metrics(train_losses, val_losses, train_accs, val_accs)