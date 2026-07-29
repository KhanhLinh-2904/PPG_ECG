import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model_1branch import FocusedNeuralNetwork
import os

# Cấu hình thiết bị chạy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device for testing: {device}")

# Đường dẫn tệp tin
CHECKPOINT_PATH = "/home/linhhima/PPG_ECG/AF_Detection/new_checkpoints_1branch/best_model.pth"
TEST_DATA_PATH = "/home/linhhima/PPG_ECG/AF_Detection/total_mimic_af_flow_segment_1.npz" # Bạn kiểm tra lại tên file test nhé

def test_model(test_loader, model, device, threshold=0.5):
    model.to(device)
    model.eval() # Chuyển mô hình sang chế độ đánh giá (tắt Dropout, BatchNorm cố định)

    all_preds = []
    all_labels = []

    with torch.no_grad(): # Tắt tính toán gradient để giải phóng bộ nhớ và tăng tốc
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            
            # Forward pass: nhận đầu ra dạng xác suất [batch_size, 1]
            outputs = model(inputs)
            
            # Áp dụng ngưỡng tối ưu Youden Index để phân định lớp AF (1) hay Non-AF (0)
            predicted = (outputs >= threshold).float()
            
            # Lưu lại kết quả dự đoán và nhãn gốc (chuyển về CPU để tính toán bằng NumPy)
            all_preds.extend(predicted.cpu().numpy().flatten())
            all_labels.extend(labels.numpy().flatten())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # --- TÍNH TOÁN CÁC THÔNG SỐ CỦA CONFUSION MATRIX ---
    # Lớp 1: AF, Lớp 0: Non-AF
    TP = np.sum((all_preds == 1) & (all_labels == 1))
    TN = np.sum((all_preds == 0) & (all_labels == 0))
    FP = np.sum((all_preds == 1) & (all_labels == 0))
    FN = np.sum((all_preds == 0) & (all_labels == 1))

    # --- TÍNH TOÁN CÁC CHỈ SỐ ĐÁNH GIÁ (Bảo vệ lỗi chia cho 0 bằng 1e-7) ---
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0  # Hay còn gọi là Sensitivity
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # --- HIỂN THỊ KẾT QUẢ ---
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
    # 1. Tải dữ liệu kiểm thử (Test Data)
    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu test tại: {TEST_DATA_PATH}")
        
    test_data = np.load(TEST_DATA_PATH)
    X_test = torch.tensor(test_data['X'], dtype=torch.float32)
    y_test = torch.tensor(test_data['y'], dtype=torch.float32) # Giữ dạng float cho đồng bộ

    print(f"Shape of X_test: {X_test.shape}")
    print(f"Shape of y_test: {y_test.shape}")

    # 2. Tạo DataLoader cho tập test
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False) # Tập test giữ shuffle=False

    # 3. Khởi tạo mô hình cấu trúc 1 nhánh
    model = FocusedNeuralNetwork()

    # 4. Tải trọng số từ file checkpoint đã lưu
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
        # Nạp trọng số và map trực tiếp vào thiết bị xử lý (CPU/GPU)
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    else:
        raise FileNotFoundError(f"Không tìm thấy tệp checkpoint tại: {CHECKPOINT_PATH}. Vui lòng chạy train trước!")

    # 5. Chạy đánh giá và xuất báo cáo kết quả
    # Bạn có thể thay đổi tham số threshold nếu muốn khảo sát các ngưỡng khác
    test_model(test_loader, model, device, threshold=0.5)