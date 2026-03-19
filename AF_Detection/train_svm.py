import numpy as np
import os
import random
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ----------- Cấu hình hệ thống -----------
CHECKPOINT_DIR = "AF_Detection/checkpoints_SVM"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

if __name__ == "__main__":
    set_seed(42)
    print("Bắt đầu tải dữ liệu...")

    # 1. Load training data (Giữ nguyên dạng Numpy, không dùng Torch Tensor)
    train_data = np.load('AF_Detection/detect_af_MIT_BIH_train.npz')
    X_train = train_data['X']
    y_train = train_data['y']

    # 2. Chuẩn hóa dữ liệu (CỰC KỲ QUAN TRỌNG CHO SVM RBF)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 3. Khởi tạo mô hình SVM với RBF Kernel
    print("Khởi tạo và huấn luyện mô hình SVM (RBF Kernel)...")
    svm_model = SVC(
        kernel='rbf', 
        C=1.0,                    # Độ nới lỏng biên (Regularization)
        gamma='scale',            # Hệ số cong của mặt phẳng RBF
        class_weight='balanced',  # Tự động xử lý mất cân bằng AF / Non-AF
        probability=True,         # Bật tính năng xuất ra xác suất (để tính AUROC sau này)
        random_state=42
    )

    # 4. Huấn luyện (Chỉ mất đúng 1 dòng code, không cần vòng lặp Epoch)
    svm_model.fit(X_train_scaled, y_train)

    # 5. Đánh giá nhanh trên chính tập Train
    y_pred = svm_model.predict(X_train_scaled)
    train_acc = accuracy_score(y_train, y_pred) * 100
    
    print("-" * 40)
    print(f"Hoàn thành Huấn luyện!")
    print(f"Train Accuracy: {train_acc:.2f}%")
    print("\nChi tiết (Classification Report):")
    print(classification_report(y_train, y_pred, target_names=['Non-AF', 'AF']))
    print("-" * 40)

    # 6. Lưu mô hình và bộ chuẩn hóa
    # PHẢI LƯU SCALER để sau này đem ra chuẩn hóa tập Test
    model_path = os.path.join(CHECKPOINT_DIR, "svm_rbf_best_model.pkl")
    scaler_path = os.path.join(CHECKPOINT_DIR, "svm_scaler.pkl")
    
    joblib.dump(svm_model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"Đã lưu mô hình SVM tại: {model_path}")
    print(f"Đã lưu bộ chuẩn hóa tại: {scaler_path}")

    # (Lưu ý: SVM không có khái niệm epochs, nên hàm plot_metrics vẽ biểu đồ Loss/Acc qua thời gian đã được gỡ bỏ)