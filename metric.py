import numpy as np

def calculate_metrics(y_true, y_pred):
    """
    Tính toán rRMSE và Pearson Correlation Coefficient.
    
    Tham số:
    y_true (np.array): Vector giá trị thực tế (Ground truth)
    y_pred (np.array): Vector giá trị dự báo (Reconstructed/Predicted)
    """
    
    # Đảm bảo dữ liệu là mảng numpy
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # 1. Tính rRMSE (Relative Root Mean-Squared Error)
    # Công thức: ||y_true - y_pred||2 / ||y_true||2
    error_vector = y_true - y_pred
    rRMSE = np.linalg.norm(error_vector, ord=2) / np.linalg.norm(y_true, ord=2)
    
    # 2. Tính Pearson’s Correlation Coefficient (rho)
    # Công thức: (y_true - mean_true).T @ (y_pred - mean_pred) / (norm_diff_true * norm_diff_pred)
    y_true_centered = y_true - np.mean(y_true)
    y_pred_centered = y_pred - np.mean(y_pred)
    
    numerator = np.dot(y_true_centered, y_pred_centered)
    denominator = np.linalg.norm(y_true_centered, ord=2) * np.linalg.norm(y_pred_centered, ord=2)
    
    rho = numerator / denominator
    
    return rRMSE, rho

if __name__ == "__main__":
    # --- Ví dụ sử dụng ---
    # Giả sử đây là tín hiệu ECG thực tế và tín hiệu tái tạo
    y_test = [0.1, 0.2, 0.5, 1.2, 0.5, 0.2, 0.1]
    y_hat  = [0.12, 0.18, 0.48, 1.15, 0.52, 0.21, 0.09]

    rrmse_val, rho_val = calculate_metrics(y_test, y_hat)

    print(f"rRMSE: {rrmse_val:.4f}")
    print(f"Pearson Correlation (rho): {rho_val:.4f}")