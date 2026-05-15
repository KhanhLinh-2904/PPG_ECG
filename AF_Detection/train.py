import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import NeuralNetwork
import os
import matplotlib.pyplot as plt
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using devices: {device}")
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
CHECKPOINT_DIR = "/home/linhhima/PPG_ECG/AF_Detection/new_checkpoints/"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def train_model(train_loader, epochs, model, criterion, optimizer, device):
    train_losses = []
    train_accuracies = []

    best_train_acc = 0.0
    best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for inputs, labels in train_loader:
            # Forward pass
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass & Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
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

    print(f"\nComplete! Model is good at {best_train_acc:.2f}% Train Accuracy tại epoch {epoch_best}.")
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
    plt.savefig("training_results.png")
    plt.show()


if __name__ == "__main__":
    set_seed(42)
    # 1. Load training data
    train_data = np.load('/home/linhhima/PPG_ECG/AF_Detection/detect_af_MIT_BIH_train.npz')
    # Note: Adjust keys 'X', 'y' to match your .npz file
    X_train = torch.tensor(train_data['X'], dtype=torch.float32)
    y_train = torch.tensor(train_data['y'], dtype=torch.long)

    # 2. DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 3. Initialize Model, Loss, Optimizer
    num_classes = len(torch.unique(y_train))
    model = NeuralNetwork(num_classes=num_classes)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 200

    # 4. Train
    train_losses, train_accs = train_model(
        train_loader, epochs, model, criterion, optimizer, device
    )

    # 5. Plot results
    plot_metrics(train_losses, train_accs)