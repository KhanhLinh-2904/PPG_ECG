import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model_1branch import FocusedNeuralNetwork
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import os
import matplotlib.pyplot as plt

# Cấu hình thiết bị chạy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Đường dẫn tệp tin
CHECKPOINT_PATH = "/home/linhhima/PPG_ECG/AF_Detection/new_checkpoints_1branch/best_model.pth"
TEST_DATA_PATH = "/home/linhhima/PPG_ECG/AF_Detection/total_mimic_af_ppg2ecg_segment_1.npz"

def get_model_probabilities(data_loader, model, device):
    """ Hàm thu thập tất cả các giá trị xác suất (raw probabilities từ Sigmoid) và nhãn gốc """
    model.to(device)
    model.eval()
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs) # Xác suất từ Sigmoid [batch_size, 1]
            
            all_probs.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(labels.numpy().flatten())
            
    return np.array(all_probs), np.array(all_labels)

def find_optimal_youden_threshold(probs, labels):
    """ Hàm quét qua các ngưỡng để tìm Youden Index lớn nhất trên tập 80% """
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
    """ Hàm tính toán ma trận nhầm lẫn và các chỉ số từ mảng xác suất và ngưỡng cho trước """
    preds = (probs >= threshold).astype(float)

    TP = np.sum((preds == 1) & (labels == 1))
    TN = np.sum((preds == 0) & (labels == 0))
    FP = np.sum((preds == 1) & (labels == 0))
    FN = np.sum((preds == 0) & (labels == 1))

    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0  # Sensitivity
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


# --- ĐÃ THÊM: HÀM VẼ ĐỒ THỊ ROC GIỐNG HỆT HÌNH MẪU ---
def plot_roc_curve(select_labels, select_probs, test_labels, test_probs, optimal_threshold):
    """ Hàm vẽ và lưu đồ thị ROC Curve Split-Testing Analysis khớp 100% định dạng mẫu """
    # Tính toán ROC và AUC cho hai tập
    fpr_learn, tpr_learn, _ = roc_curve(select_labels, select_probs)
    auc_learn = auc(fpr_learn, tpr_learn)

    fpr_test, tpr_test, thresholds_test = roc_curve(test_labels, test_probs)
    auc_test = auc(fpr_test, tpr_test)

    # Tìm điểm tọa độ ứng với optimal_threshold trên tập test để chấm đỏ
    idx = np.argmin(np.abs(thresholds_test - optimal_threshold))
    fpr_point = fpr_test[idx]
    tpr_point = tpr_test[idx]

    # Khởi tạo khung vẽ
    plt.figure(figsize=(8, 6.5), dpi=100)
    plt.grid(True, linestyle='-', alpha=0.3)

    # 1. Đường Learning ROC: nét đứt màu xanh dodgerblue
    plt.plot(fpr_learn, tpr_learn, color='#1e90ff', linestyle='--', linewidth=2, 
             label=f'Learning ROC Curve (AUC = {auc_learn:.4f})')

    # 2. Đường Testing ROC: nét liền màu cam đậm
    plt.plot(fpr_test, tpr_test, color='#ff8c00', linestyle='-', linewidth=2.5, 
             label=f'Testing ROC Curve (AUC = {auc_test:.4f})')

    # 3. Đường Chance Line: nét đứt màu xanh navy
    plt.plot([0, 1], [0, 1], color='#000080', linestyle='--', linewidth=1.5, label='Chance Line')

    # 4. Chấm điểm ngưỡng tối ưu: màu đỏ tròn to
    plt.plot(fpr_point, tpr_point, 'ro', markersize=11, 
             label=f'Applied Threshold ({optimal_threshold:.4f})\n(FPR: {fpr_point:.2f}, TPR: {tpr_point:.2f})')

    # Giới hạn trục đồ thị và dán nhãn
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('ROC Curve Split-Testing Analysis', fontsize=13, fontweight='bold', pad=12)

    # Đặt Legend ở góc dưới phải, có phủ nền mờ đè lưới
    plt.legend(loc='lower right', fontsize=10, frameon=True, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig("ROC_Curve_Split_Testing.png", dpi=300)
    print("Graph saved successfully as 'ROC_Curve_Split_Testing.png'!")
    plt.show()


if __name__ == "__main__":
    # 1. Tải toàn bộ dữ liệu MIMIC AF
    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {TEST_DATA_PATH}")
        
    total_data = np.load(TEST_DATA_PATH)
    X_total = total_data['X']
    y_total = total_data['y']
    
    print(f"Total MIMIC AF dataset shape: {X_total.shape}")

    # 2. Chia tập dữ liệu: 80% tìm ngưỡng (Selection) và 20% kiểm thử (Testing)
    X_select, X_test, y_select, y_test = train_test_split(
        X_total, y_total, test_size=0.20, random_state=41, stratify=y_total
    )
    print(f"Selection set (80%) shape: {X_select.shape}")
    print(f"Testing set (20%) shape: {X_test.shape}")

    # Chuyển đổi sang PyTorch Tensors
    X_select_t = torch.tensor(X_select, dtype=torch.float32)
    y_select_t = torch.tensor(y_select, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    # Khởi tạo DataLoaders
    select_loader = DataLoader(TensorDataset(X_select_t, y_select_t), batch_size=64, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=64, shuffle=False)

    # 3. Khởi tạo mô hình và nạp trọng số checkpoint
    model = FocusedNeuralNetwork()
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    else:
        raise FileNotFoundError(f"Không tìm thấy tệp checkpoint tại: {CHECKPOINT_PATH}")

    # 4. Trích xuất xác suất dự đoán (probabilities) cho cả 2 tập dữ liệu
    select_probs, select_labels = get_model_probabilities(select_loader, model, device)
    test_probs, test_labels = get_model_probabilities(test_loader, model, device)

    # 5. Tìm ngưỡng tối ưu trên tập Selection (80%)
    optimal_threshold = find_optimal_youden_threshold(select_probs, select_labels)

    # 6. Đánh giá kiểm thử độc lập trên tập Testing (20%) bằng ngưỡng vừa tìm được
    evaluate_performance(test_probs, test_labels, threshold=optimal_threshold, dataset_name="Testing Set (20%)")
    
    # --- ĐÃ THÊM: KÍCH HOẠT VẼ ĐỒ THỊ ---
    plot_roc_curve(select_labels, select_probs, test_labels, test_probs, optimal_threshold)