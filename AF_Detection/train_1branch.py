import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model_1branch import FocusedNeuralNetwork 
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

CHECKPOINT_DIR = "/home/linhhima/PPG_ECG/AF_Detection/checkpoints_1branch/"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def train_model(train_loader, epochs, model, criterion, optimizer, device):
    train_losses = []
    train_accuracies = []

    best_train_acc = 0.0
    best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    epoch_best = 0

    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            
            predicted = (outputs >= 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100 * correct / total

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        print(f"Epoch {epoch+1:03d}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1:03d}.pth")
        torch.save(model.state_dict(), checkpoint_path)

        if train_acc > best_train_acc:
            best_train_acc = train_acc
            epoch_best = epoch + 1
            torch.save(model.state_dict(), best_model_path)

    print(f"\nComplete! Best Model at {best_train_acc:.2f}% Train Accuracy at epoch {epoch_best}.")
    return train_losses, train_accuracies

def plot_metrics(train_losses, train_accs):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-o', label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'r-o', label='Train Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_results_1branch.png")
    plt.show()

if __name__ == "__main__":
    set_seed(42)
    
    train_data = np.load('/home/linhhima/PPG_ECG/AF_Detection/detect_af_MIT_BIH_train.npz')
    
    X_train = torch.tensor(train_data['X'], dtype=torch.float32)
    y_train = torch.tensor(train_data['y'], dtype=torch.float32).unsqueeze(1)

    print(f"Shape of X_train (Must be [Batch, 4]): {X_train.shape}") 
    print(f"Shape of y_train (Must be [Batch, 1]): {y_train.shape}")

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = FocusedNeuralNetwork()
    
    criterion = nn.BCELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 200

    train_losses, train_accs = train_model(
        train_loader, epochs, model, criterion, optimizer, device
    )

    plot_metrics(train_losses, train_accs)