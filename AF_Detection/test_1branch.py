import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model_1branch import FocusedNeuralNetwork
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device for testing: {device}")

CHECKPOINT_PATH = "/home/linhhima/PPG_ECG/AF_Detection/checkpoints_1branch/best_model.pth"
TEST_DATA_PATH = "/home/linhhima/PPG_ECG/AF_Detection/total_mimic_af_flow_segment_1.npz"

def test_model(test_loader, model, device, threshold=0.5):
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            
            outputs = model(inputs)
            
            predicted = (outputs >= threshold).float()
            
            all_preds.extend(predicted.cpu().numpy().flatten())
            all_labels.extend(labels.numpy().flatten())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    TP = np.sum((all_preds == 1) & (all_labels == 1))
    TN = np.sum((all_preds == 0) & (all_labels == 0))
    FP = np.sum((all_preds == 1) & (all_labels == 0))
    FN = np.sum((all_preds == 0) & (all_labels == 1))

    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "="*50)
    print(f"      AF DETECTION TEST RESULTS (Threshold: {threshold})")
    print("="*50)
    print(f"True Positives  (TP) : {TP}")
    print(f"True Negatives  (TN) : {TN}")
    print(f"False Positives (FP) : {FP}")
    print(f"False Negatives (FN) : {FN}")
    print("-"*50)
    print(f"Accuracy            : {accuracy * 100:.2f}%")
    print(f"Precision           : {precision * 100:.2f}%")
    print(f"Recall (Sensitivity): {recall * 100:.2f}%")
    print(f"Specificity         : {specificity * 100:.2f}%")
    print(f"F1-Score            : {f1_score * 100:.2f}%")
    print("="*50 + "\n")

    return accuracy, precision, recall, specificity

if __name__ == "__main__":
    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(f"Test data file not found at: {TEST_DATA_PATH}")
        
    test_data = np.load(TEST_DATA_PATH)
    X_test = torch.tensor(test_data['X'], dtype=torch.float32)
    y_test = torch.tensor(test_data['y'], dtype=torch.float32)

    print(f"Shape of X_test: {X_test.shape}")
    print(f"Shape of y_test: {y_test.shape}")

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = FocusedNeuralNetwork()

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    else:
        raise FileNotFoundError(f"Checkpoint file not found at: {CHECKPOINT_PATH}. Please train the model first!")

    test_model(test_loader, model, device, threshold=0.5)