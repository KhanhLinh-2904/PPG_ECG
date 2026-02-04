import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import NeuralNetwork
import os
import matplotlib.pyplot as plt
import random

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Nếu dùng nhiều GPU
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
# Thư mục lưu trữ checkpoint
CHECKPOINT_DIR = "AF_Detection/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def train_model(train_loader, epochs, model, criterion, optimizer):
    train_losses = []
    train_accuracies = []

    best_train_acc = 0.0
    best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for inputs, labels in train_loader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass & Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Thống kê
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100 * correct / total

        # Lưu lịch sử
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        print(f"Epoch {epoch+1:03d}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

        # Lưu checkpoint hàng kỳ (mỗi epoch)
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1:03d}.pth")
        torch.save(model.state_dict(), checkpoint_path)

        # Lưu model tốt nhất dựa trên Train Accuracy
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            epoch_best = epoch + 1
            torch.save(model.state_dict(), best_model_path)

    print(f"\nKết thúc! Model tốt nhất đạt {best_train_acc:.2f}% Train Accuracy tại epoch {epoch_best}.")
    return train_losses, train_accuracies


def plot_metrics(train_losses, train_accs):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    # Biểu đồ Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-o', label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    # Biểu đồ Accuracy
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
    # 1. Load dữ liệu training
    train_data = np.load('AF_Detection/detect_af_MIMIC_train_mm.npz')
    # Lưu ý: Chỉnh sửa key 'X', 'y' cho đúng với file .npz của bạn
    X_train = torch.tensor(train_data['X'], dtype=torch.float32)
    y_train = torch.tensor(train_data['y'], dtype=torch.long)

    # 2. Tạo DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 3. Khởi tạo Model, Loss, Optimizer
    num_classes = len(torch.unique(y_train))
    model = NeuralNetwork(num_classes=num_classes)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 200

    # 4. Huấn luyện
    train_losses, train_accs = train_model(
        train_loader, epochs, model, criterion, optimizer
    )

    # 5. Vẽ biểu đồ kết quả
    plot_metrics(train_losses, train_accs)