import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import FocusedNeuralNetwork
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# ----------- Configuration -----------
CHECKPOINT_PATH = "/home/linhhima/PPG_ECG/AF_Detection/new_checkpoints_mimic_af/best_model.pth"
TEST_DATA_PATH = "/home/linhhima/PPG_ECG/AF_Detection/detect_af_MIT_BIH_test.npz"

BATCH_SIZE = 32

# ĐÃ KHÓA NGƯỠNG TỐI ƯU TỪ TẬP TRAIN
LOCKED_THRESHOLD = 0.5

# ----------- Load Test Data -----------
test_data = np.load(TEST_DATA_PATH)
X_test = torch.tensor(test_data['X'], dtype=torch.float32)
y_test = torch.tensor(test_data['y'], dtype=torch.long)
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

print(f"Shape of X_test: {X_test.shape}") 

# ----------- Load Model -----------
num_classes = len(torch.unique(y_test))
assert num_classes == 2, "This code assumes binary classification (2 classes)."

model = FocusedNeuralNetwork(num_classes=num_classes)

try:
    model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))
except TypeError:
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    
model.eval()

# ----------- Evaluation (Thu thập Logits & Tính Xác Suất) -----------
criterion = nn.CrossEntropyLoss()
test_loss = 0.0
total = 0

all_labels = []
all_probs = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("[*] Đang chạy suy luận (Inference) trên tập Test...")
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device) 
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * inputs.size(0)
        total += labels.size(0)
        
        probs = torch.softmax(outputs, dim=1)[:, 1]

        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

avg_loss = test_loss / total
y_true = np.array(all_labels)
y_probs = np.array(all_probs)

# ----------- HẬU XỬ LÝ: ÁP DỤNG NGƯỠNG ĐÃ KHÓA -----------
fpr, tpr, roc_thresholds = roc_curve(y_true, y_probs)
roc_auc = auc(fpr, tpr)

print(f"\n[*] Đang sử dụng Threshold đã khóa: {LOCKED_THRESHOLD:.4f}")
# Tìm index của điểm trên đồ thị ROC gần với Threshold đã khóa nhất để lát nữa vẽ hình
operating_idx = np.argmin(np.abs(roc_thresholds - LOCKED_THRESHOLD))

# ----------- CHẨN ĐOÁN & TÍNH TOÁN METRICS -----------
# Quyết định nhãn dựa trên Ngưỡng 0.5612
y_pred = (y_probs >= LOCKED_THRESHOLD).astype(int)

# Xuất Confusion Matrix
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

accuracy = 100 * (tp + tn) / total
precision = 100 * tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Sensitivity
specificity = 100 * tn / (tn + fp) if (tn + fp) > 0 else 0.0

# ----------- IN BÁO CÁO KẾT QUẢ -----------
print("\n" + "="*60)
print(f" 📊 BÁO CÁO ĐÁNH GIÁ TRÊN TẬP TEST (THRESHOLD = {LOCKED_THRESHOLD:.4f})")
print("="*60)
print(f"Test Loss            : {avg_loss:.4f}")
print(f"AUROC                : {roc_auc:.4f}")
print("-" * 60)
print(f"True Positives (TP)  : {tp}")
print(f"True Negatives (TN)  : {tn}")
print(f"False Positives (FP) : {fp} (Báo động giả)")
print(f"False Negatives (FN) : {fn} (Bỏ lọt bệnh)")
print("-" * 60)
print(f"Accuracy             : {accuracy:.2f}%")
print(f"Precision            : {precision:.2f}%")
print(f"Recall (Sensitivity) : {recall:.2f}%")
print(f"Specificity          : {specificity:.2f}%")
print("="*60 + "\n")

# =================================================================
# VẼ ĐỒ THỊ ROC VỚI ĐIỂM HOẠT ĐỘNG (OPERATING POINT) ĐÃ KHÓA
# =================================================================
plt.figure(figsize=(9, 7))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUROC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance Line (Random Guess)')

# Lấy tọa độ của Threshold đã khóa trên đường cong ROC
op_fpr = fpr[operating_idx]
op_tpr = tpr[operating_idx]

# Vẽ điểm đánh dấu hiệu suất thực tế của mô hình trên tập Test
plt.scatter(op_fpr, op_tpr, marker='o', color='red', s=120, 
            label=f'Locked Threshold ({LOCKED_THRESHOLD:.4f})\nTest FPR: {op_fpr:.2f}, Test TPR: {op_tpr:.2f}', zorder=5)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
plt.title("Test Set: ROC Curve & Locked Operating Point", fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# =================================================================
# VẼ ĐỒ THỊ PRECISION-RECALL (PRC)
# =================================================================
precision_curve, recall_curve, prc_thresholds = precision_recall_curve(all_labels, all_probs)
avg_precision = average_precision_score(all_labels, all_probs)

plt.figure(figsize=(9, 7))
plt.plot(recall_curve, precision_curve, color='blue', lw=2,
         label=f'PRC Curve (Average Precision = {avg_precision:.4f})')
plt.xlabel('Recall (Sensitivity)', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Test Set: Precision-Recall (PRC) Curve', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(loc='lower left', fontsize=10)
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.tight_layout()
plt.show()