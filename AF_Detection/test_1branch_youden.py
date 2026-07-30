import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model_1branch import FocusedNeuralNetwork
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import os
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

CHECKPOINT_PATH = "/home/linhhima/PPG_ECG/AF_Detection/checkpoints_1branch/best_model.pth"
TEST_DATA_PATH = "/home/linhhima/PPG_ECG/AF_Detection/total_mimic_af_flow_segment_1.npz"

def get_model_probabilities(data_loader, model, device):
    model.to(device)
    model.eval()
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            all_probs.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(labels.numpy().flatten())
            
    return np.array(all_probs), np.array(all_labels)

def find_optimal_youden_threshold(probs, labels):
    thresholds = np.arange(0.001, 1.0, 0.001)
    best_threshold = 0.5
    max_youden_index = -1.0
    
    print("\n--- Optimizing Threshold via Youden Index (on 80% Selection Subset) ---")
    
    for t in thresholds:
        preds = (probs >= t).astype(float)
        
        TP = np.sum((preds == 1) & (labels == 1))
        TN = np.sum((preds == 0) & (labels == 0))
        FP = np.sum((preds == 1) & (labels == 0))
        FN = np.sum((preds == 0) & (labels == 1))
        
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        
        youden_index = sensitivity + specificity - 1
        
        if youden_index > max_youden_index:
            max_youden_index = youden_index
            best_threshold = t
            
    print(f"Optimal Threshold Found: {best_threshold:.4f} (Max Youden Index J: {max_youden_index:.4f})")
    return best_threshold

def evaluate_performance(probs, labels, threshold, dataset_name="Testing Set (20%)"):
    preds = (probs >= threshold).astype(float)

    TP = np.sum((preds == 1) & (labels == 1))
    TN = np.sum((preds == 0) & (labels == 0))
    FP = np.sum((preds == 1) & (labels == 0))
    FN = np.sum((preds == 0) & (labels == 1))

    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "="*50)
    print(f"  AF DETECTION PERFORMANCE FOR: {dataset_name.upper()}")
    print(f"  Selected Threshold: {threshold:.4f}")
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

def plot_roc_curve(select_labels, select_probs, test_labels, test_probs, optimal_threshold):
    fpr_learn, tpr_learn, _ = roc_curve(select_labels, select_probs)
    auc_learn = auc(fpr_learn, tpr_learn)

    fpr_test, tpr_test, thresholds_test = roc_curve(test_labels, test_probs)
    auc_test = auc(fpr_test, tpr_test)

    idx = np.argmin(np.abs(thresholds_test - optimal_threshold))
    fpr_point = fpr_test[idx]
    tpr_point = tpr_test[idx]

    plt.figure(figsize=(8, 6.5), dpi=100)
    plt.grid(True, linestyle='-', alpha=0.3)

    plt.plot(fpr_learn, tpr_learn, color='#1e90ff', linestyle='--', linewidth=2, 
             label=f'Learning ROC Curve (AUC = {auc_learn:.4f})')

    plt.plot(fpr_test, tpr_test, color='#ff8c00', linestyle='-', linewidth=2.5, 
             label=f'Testing ROC Curve (AUC = {auc_test:.4f})')

    plt.plot([0, 1], [0, 1], color='#000080', linestyle='--', linewidth=1.5, label='Chance Line')

    plt.plot(fpr_point, tpr_point, 'ro', markersize=11, 
             label=f'Applied Threshold ({optimal_threshold:.4f})\n(FPR: {fpr_point:.2f}, TPR: {tpr_point:.2f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('ROC Curve Split-Testing Analysis', fontsize=13, fontweight='bold', pad=12)

    plt.legend(loc='lower right', fontsize=10, frameon=True, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig("ROC_Curve_Split_Testing.png", dpi=300)
    print("Graph saved successfully as 'ROC_Curve_Split_Testing.png'!")
    plt.show()

if __name__ == "__main__":
    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(f"Data file not found at: {TEST_DATA_PATH}")
        
    total_data = np.load(TEST_DATA_PATH)
    X_total = total_data['X']
    y_total = total_data['y']
    
    print(f"Total MIMIC AF dataset shape: {X_total.shape}")

    X_select, X_test, y_select, y_test = train_test_split(
        X_total, y_total, test_size=0.20, random_state=41, stratify=y_total
    )
    print(f"Selection set (80%) shape: {X_select.shape}")
    print(f"Testing set (20%) shape: {X_test.shape}")

    X_select_t = torch.tensor(X_select, dtype=torch.float32)
    y_select_t = torch.tensor(y_select, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    select_loader = DataLoader(TensorDataset(X_select_t, y_select_t), batch_size=64, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=64, shuffle=False)

    model = FocusedNeuralNetwork()
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    else:
        raise FileNotFoundError(f"Checkpoint file not found at: {CHECKPOINT_PATH}")

    select_probs, select_labels = get_model_probabilities(select_loader, model, device)
    test_probs, test_labels = get_model_probabilities(test_loader, model, device)

    optimal_threshold = find_optimal_youden_threshold(select_probs, select_labels)

    evaluate_performance(test_probs, test_labels, threshold=optimal_threshold, dataset_name="Testing Set (20%)")
    
    plot_roc_curve(select_labels, select_probs, test_labels, test_probs, optimal_threshold)