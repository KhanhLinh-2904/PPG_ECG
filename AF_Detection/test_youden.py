import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import FocusedNeuralNetwork
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

CHECKPOINT_PATH = "/home/linhhima/PPG_ECG/AF_Detection/checkpoints/best_model.pth"
TEST_DATA_PATH = "/home/linhhima/PPG_ECG/AF_Detection/total_mimic_af_flow_segment_1.npz"
BATCH_SIZE = 32
SEED = 101

CUSTOM_THRESHOLD = None

if not os.path.exists(TEST_DATA_PATH):
    raise FileNotFoundError(f"Data file not found at: {TEST_DATA_PATH}")

test_data = np.load(TEST_DATA_PATH)
X_all = test_data['X']
y_all = test_data['y']

print(f"[*] Total initial samples: {X_all.shape[0]} segments.")

stratify_labels = (y_all >= 0.5).astype(int)

X_learn, X_test, y_learn, y_test = train_test_split(
    X_all, y_all, 
    test_size=0.50,
    random_state=SEED, 
    stratify=stratify_labels
)

X_learn_t = torch.tensor(X_learn, dtype=torch.float32)
y_learn_t = torch.tensor(y_learn, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

learn_loader = DataLoader(TensorDataset(X_learn_t, y_learn_t), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False)

print(f" -> Learning set size: {X_learn.shape[0]} samples")
print(f" -> Testing set size: {X_test.shape[0]} samples")

model = FocusedNeuralNetwork()

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(f"Model checkpoint not found at: {CHECKPOINT_PATH}")

try:
    model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))
except TypeError:
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def inference_pipeline(dataloader, desc="Inference"):
    all_labels = []
    all_probs = []
    print(f"[*] Running process: {desc}...")
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs).view(-1)
            
            all_labels.extend(labels.view(-1).numpy())
            all_probs.extend(outputs.cpu().numpy())
    return np.array(all_labels), np.array(all_probs)

y_true_learn, y_probs_learn = inference_pipeline(learn_loader, desc="Learning Subset")
y_true_test, y_probs_test = inference_pipeline(test_loader, desc="Testing Subset")

y_true_learn_binary = (y_true_learn >= 0.5).astype(int)
y_true_test_binary = (y_true_test >= 0.5).astype(int)

fpr_learn, tpr_learn, roc_thresholds_learn = roc_curve(y_true_learn_binary, y_probs_learn)
roc_auc_learn = auc(fpr_learn, tpr_learn)

if CUSTOM_THRESHOLD is None:
    print("\n[*] Optimizing optimal cut-off threshold on learning set via Youden's Index...")
    youden_j_learn = tpr_learn - fpr_learn
    best_idx_learn = np.argmax(youden_j_learn)
    
    optimal_threshold = roc_thresholds_learn[best_idx_learn]
    max_j_learn = youden_j_learn[best_idx_learn]
    
    print(f" 🎯 Threshold optimization results:")
    print(f"    - Theoretical Max Youden's J: {max_j_learn:.4f}")
    print(f"    - Found optimal threshold   : {optimal_threshold:.4f}")
else:
    optimal_threshold = CUSTOM_THRESHOLD
    print(f"\n[*] Applying fixed specified threshold: {optimal_threshold:.4f}")

y_pred_test_binary = (y_probs_test >= optimal_threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(y_true_test_binary, y_pred_test_binary).ravel()
total_test = len(y_true_test_binary)

accuracy = 100 * (tp + tn) / total_test
precision = 100 * tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0.0
specificity = 100 * tn / (tn + fp) if (tn + fp) > 0 else 0.0

fpr_test, tpr_test, roc_thresholds_test = roc_curve(y_true_test_binary, y_probs_test)
roc_auc_test = auc(fpr_test, tpr_test)

print("\n" + "="*60)
print(f" 🏆 INDEPENDENT EXPERIMENTAL REPORT ON TESTING SET (DATASET)")
print(f"    Applied threshold learned from train set: THRESHOLD = {optimal_threshold:.4f}")
print("="*60)
print(f"AUROC (Learning set) : {roc_auc_learn:.4f}")
print(f"AUROC (Testing set)  : {roc_auc_test:.4f}")
print("-" * 60)
print(f"True Positives (TP)        : {tp}")
print(f"True Negatives (TN)        : {tn}")
print(f"False Positives (FP)       : {fp} (False alarms)")
print(f"False Negatives (FN)       : {fn} (Missed AF cases)")
print("-" * 60)
print(f"Accuracy                   : {accuracy:.2f}%")
print(f"Precision                  : {precision:.2f}%")
print(f"Recall (Sensitivity)       : {recall:.2f}%")
print(f"Specificity                : {specificity:.2f}%")
print("="*60 + "\n")

plt.figure(figsize=(9, 7))

plt.plot(fpr_learn, tpr_learn, color='dodgerblue', lw=2, 
         linestyle='--', label=f'Learning ROC Curve (AUC = {roc_auc_learn:.4f})')
plt.plot(fpr_test, tpr_test, color='darkorange', lw=2.5, label=f'Testing ROC Curve (AUC = {roc_auc_test:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Chance Line')

applied_fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
applied_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

plt.scatter(applied_fpr, applied_tpr, marker='o', color='red', s=130, 
            label=f'Applied Threshold ({optimal_threshold:.4f})\n(FPR: {applied_fpr:.2f}, TPR: {applied_tpr:.2f})', zorder=5)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
plt.title("ROC Curve Split-Testing Analysis", fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

precision_curve, recall_curve, _ = precision_recall_curve(y_true_test_binary, y_probs_test)
avg_precision = average_precision_score(y_true_test_binary, y_probs_test)

plt.figure(figsize=(9, 7))
plt.plot(recall_curve, precision_curve, color='crimson', lw=2.5, 
         label=f'Testing PRC Curve (Average Precision = {avg_precision:.4f})')
plt.xlabel('Recall (Sensitivity)', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall (PRC) Curve (Testing Set)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(loc='lower left', fontsize=10)
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.tight_layout()
plt.show()