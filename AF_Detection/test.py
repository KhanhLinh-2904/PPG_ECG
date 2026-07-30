import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import FocusedNeuralNetwork
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/home/linhhima/PPG_ECG/AF_Detection/checkpoints/best_model.pth"
TEST_DATA_PATH = "/home/linhhima/PPG_ECG/AF_Detection/total_mimic_af.npz"

print(f"Using device: {device}")

if __name__ == "__main__":
    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(f"Test data file not found at: {TEST_DATA_PATH}")
        
    test_data = np.load(TEST_DATA_PATH)
    X_test = torch.tensor(test_data['X'], dtype=torch.float32)
    y_test = torch.tensor(test_data['y'], dtype=torch.float32)

    print(f"Shape of X_test: {X_test.shape}")
    print(f"Shape of y_test: {y_test.shape}")

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = FocusedNeuralNetwork()
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found at: {MODEL_PATH}")
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            all_outputs.append(outputs.cpu().numpy())
            all_labels.append(labels.numpy())

    y_pred_continuous = np.concatenate(all_outputs)
    y_true_continuous = np.concatenate(all_labels)

    DECISION_THRESHOLD = 0.5
    
    y_pred_binary = (y_pred_continuous > DECISION_THRESHOLD).astype(int)
    y_true_binary = (y_true_continuous > DECISION_THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()

    accuracy = accuracy_score(y_true_binary, y_pred_binary) * 100
    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0) * 100
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0) * 100
    sensitivity = recall 
    specificity = (tn / (tn + fp)) * 100 if (tn + fp) > 0 else 0.0

    print("\n" + "="*50)
    print("        THRESHOLD REGRESSION MODEL TEST REPORT        ")
    print("="*50)
    print(f"Decision Threshold Applied : {DECISION_THRESHOLD}")
    print(f"Confusion Matrix           : TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("-"*50)
    print(f"Accuracy                   : {accuracy:.2f}%")
    print(f"Precision                  : {precision:.2f}%")
    print(f"Sensitivity (Recall)       : {sensitivity:.2f}%")
    print(f"Specificity                : {specificity:.2f}%")
    print("="*50)

    mae_regression = np.mean(np.abs(y_pred_continuous - y_true_continuous))
    print(f"Regression Mean Absolute Error (MAE): {mae_regression:.4f}")