import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import FocusedNeuralNetwork
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import os

# 1. Cấu hình thiết bị và đường dẫn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/home/linhhima/PPG_ECG/AF_Detection/new_checkpoint_threshold/best_model.pth"
TEST_DATA_PATH = "/home/linhhima/PPG_ECG/AF_Detection/total_mimic_af_flow_segment_mse_1.npz" # Thay tên file test chuẩn của bạn

print(f"Using device: {device}")

if __name__ == "__main__":
    # 2. Tải dữ liệu kiểm thử (Test Data)
    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu test tại: {TEST_DATA_PATH}")
        
    test_data = np.load(TEST_DATA_PATH)
    X_test = torch.tensor(test_data['X'], dtype=torch.float32)
    y_test = torch.tensor(test_data['y'], dtype=torch.float32) # Nhãn liên tục (ngưỡng)

    print(f"Shape of X_test: {X_test.shape}")
    print(f"Shape of y_test: {y_test.shape}")

    # 3. Khởi tạo DataLoader cho tập Test
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 4. Tải mô hình đã huấn luyện tốt nhất
    model = FocusedNeuralNetwork()
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Không tìm thấy trọng số mô hình tại: {MODEL_PATH}")
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # 5. Tiến hành dự đoán trên toàn bộ tập Test
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs) # Đầu ra dự đoán dạng [B, 1]
            
            all_outputs.append(outputs.cpu().numpy())
            all_labels.append(labels.numpy())

    # Gộp các batch lại thành mảng phẳng 1D
    y_pred_continuous = np.concatenate(all_outputs)
    y_true_continuous = np.concatenate(all_labels)

    # 6. NHỊ PHÂN HÓA DỮ LIỆU (Chuyển từ liên tục sang Phân lớp dựa trên ngưỡng 0.5)
    DECISION_THRESHOLD = 0.5
    
    y_pred_binary = (y_pred_continuous > DECISION_THRESHOLD).astype(int)
    y_true_binary = (y_true_continuous > DECISION_THRESHOLD).astype(int)

    # 7. Tính toán các chỉ số đánh giá bằng Ma trận nhầm lẫn (Confusion Matrix)
    # tn: True Negative, fp: False Positive, fn: False Negative, tp: True Positive
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()

    # Tính toán các chỉ số yêu cầu
    accuracy = accuracy_score(y_true_binary, y_pred_binary) * 100
    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0) * 100
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0) * 100  # Recall chính là Sensitivity
    sensitivity = recall 
    specificity = (tn / (tn + fp)) * 100 if (tn + fp) > 0 else 0.0

    # 8. In kết quả báo cáo học thuật
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

    # Tính toán thêm lỗi hồi quy phụ trợ (để bạn biết mô hình dự đoán lệch bao nhiêu)
    mae_regression = np.mean(np.abs(y_pred_continuous - y_true_continuous))
    print(f"Regression Mean Absolute Error (MAE): {mae_regression:.4f}")