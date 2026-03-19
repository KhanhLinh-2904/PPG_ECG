import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, accuracy_score, log_loss
)

# ----------- Configuration -----------
# Đường dẫn tới mô hình SVM và bộ chuẩn hóa Scaler
CHECKPOINT_PATH = "AF_Detection/checkpoints_SVM/svm_rbf_best_model.pkl"
SCALER_PATH = "AF_Detection/checkpoints_SVM/svm_scaler.pkl"

# TEST_DATA_PATH = "AF_Detection/detect_af_MIT_BIH_test.npz"
TEST_DATA_PATH = "AF_Detection/detect_af_deepbeat.npz"
# TEST_DATA_PATH = "AF_Detection/detect_af_MIMIC_AF.npz"

# ----------- Load Test Data -----------
# Dữ liệu giữ nguyên dạng Numpy, không cần PyTorch Tensor hay DataLoader
print("Loading data...")
test_data = np.load(TEST_DATA_PATH)
X_test = test_data['X']
y_test = test_data['y']

# ----------- Load Model & Scaler -----------
print("Loading SVM model and Scaler...")
model = joblib.load(CHECKPOINT_PATH)
scaler = joblib.load(SCALER_PATH)

# ----------- Preprocessing -----------
# CHÚ Ý: Dùng transform (không dùng fit_transform) để giữ nguyên hệ quy chiếu của tập Train
X_test_scaled = scaler.transform(X_test)

# ----------- Evaluation & Inference -----------
print("Running inference...")
# Thay vì vòng lặp for từng batch, SVM tính toán toàn bộ ma trận trong 1 lần
probs = model.predict_proba(X_test_scaled)[:, 1]  # Xác suất nhãn 1 (AF)
preds = model.predict(X_test_scaled)              # Nhãn dự đoán tuyệt đối (0 hoặc 1)

# Tính toán Test Loss bằng LogLoss (tương đương CrossEntropy của PyTorch)
test_loss = log_loss(y_test, probs)

# Trích xuất TP, TN, FP, FN nhanh chóng bằng confusion_matrix
tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

# Metrics calculation
total = len(y_test)
accuracy = accuracy_score(y_test, preds) * 100
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Sensitivity
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

# ----------- Results -----------
print("-" * 30)
print(f"Test Loss (LogLoss): {test_loss:.4f}")
print(f"Test Accuracy: {accuracy:.2f}%")
print(f"True Positives (TP): {tp}")
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"Precision: {precision:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print("-" * 30)

# ----------- Compute ROC curve and AUROC -----------
fpr, tpr, thresholds = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)

# Plotting ROC
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUROC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('Receiver Operating Characteristic (ROC) Curve - SVM RBF')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# ----------- Compute PRC and Average Precision -----------
precision_curve, recall_curve, prc_thresholds = precision_recall_curve(y_test, probs)
avg_precision = average_precision_score(y_test, probs)

# ----------- Plot Precision-Recall Curve -----------
plt.figure()
plt.plot(recall_curve, precision_curve, color='blue', lw=2,
         label=f'Average Precision (AP) = {avg_precision:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall (PRC) Curve - SVM RBF')
plt.grid(True)
plt.legend(loc='lower left')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.show()