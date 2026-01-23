import numpy as np
from scipy.spatial.distance import euclidean
from scipy.ndimage import uniform_filter, gaussian_filter
from numba import njit
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
    # print("Error vector:", error_vector)
    rRMSE = np.linalg.norm(error_vector, ord=2) / np.linalg.norm(y_true, ord=2)
    # print("rRMSE calculation:", np.linalg.norm(error_vector, ord=2), np.linalg.norm(y_true, ord=2))
    # 2. Tính Pearson’s Correlation Coefficient (rho)
    # Công thức: (y_true - mean_true).T @ (y_pred - mean_pred) / (norm_diff_true * norm_diff_pred)
    y_true_centered = y_true - np.mean(y_true)
    y_pred_centered = y_pred - np.mean(y_pred)

    numerator = np.dot(y_true_centered, y_pred_centered)
    denominator = np.linalg.norm(y_true_centered, ord=2) * np.linalg.norm(y_pred_centered, ord=2)

    rho = numerator / denominator
    
    return rRMSE, rho

@njit
def calculate_dtw_distance(y_true, y_pred, window=100):
    """
    Tối ưu hóa DTW cho tín hiệu dài > 2400 mẫu.
    - Sử dụng @njit để biên dịch mã sang máy (tốc độ cực nhanh).
    - Sử dụng Sakoe-Chiba Band để giới hạn vùng tính toán.
    """
    # 1. Tiền xử lý dữ liệu để tránh lỗi 'setting an array element with a sequence'
    # Lưu ý: Squeeze/Flatten nên làm bên ngoài hàm njit hoặc dùng np.reshape
    y_t = np.asarray(y_true).reshape(-1)
    y_p = np.asarray(y_pred).reshape(-1)
    
    n = len(y_t)
    m = len(y_p)
    
    # Độ rộng cửa sổ Sakoe-Chiba (đảm bảo ít nhất bằng chênh lệch độ dài)
    w = max(window, abs(n - m))
    
    # 2. Khởi tạo ma trận chi phí (Dùng np.full để tối ưu bộ nhớ)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0.0

    # 3. Tính toán với ràng buộc cửa sổ
    for i in range(1, n + 1):
        # Chỉ tính toán trong phạm vi [i-w, i+w]
        start = max(1, i - w)
        end = min(m + 1, i + w + 1)
        
        for j in range(start, end):
            cost = abs(y_t[i-1] - y_p[j-1])
            
            # Quy hoạch động: Tìm đường khớp tối ưu
            last_min = min(dtw_matrix[i-1, j],    # Insertion
                           dtw_matrix[i, j-1],    # Deletion
                           dtw_matrix[i-1, j-1])  # Match
            
            dtw_matrix[i, j] = cost + last_min

    return dtw_matrix[n, m]

def calculate_cosine_similarity(y_true, y_pred):
    """
    Tính Cosine Similarity giữa hai tín hiệu.
    Kết quả gần 1.0 nghĩa là hình dạng cực kỳ giống nhau.
    """
    # Đảm bảo dữ liệu là mảng 1D phẳng
    y_t = np.asanyarray(y_true).flatten()
    y_p = np.asanyarray(y_pred).flatten()
    
    # Tính tích vô hướng (Dot product)
    dot_product = np.dot(y_t, y_p)
    
    # Tính chuẩn L2 (Magnitude)
    norm_t = np.linalg.norm(y_t)
    norm_p = np.linalg.norm(y_p)
    
    # Tránh chia cho 0
    if norm_t == 0 or norm_p == 0:
        return 0.0
        
    return dot_product / (norm_t * norm_p)
if __name__ == "__main__":
    # --- Ví dụ sử dụng ---
    # Giả sử đây là tín hiệu ECG thực tế và tín hiệu tái tạo
    y_test = [0.1, 0.2, 0.5, 1.2, 0.5, 0.2, 0.1]
    y_hat  = [0.12, 0.18, 0.48, 1.15, 0.52, 0.21, 0.09]

    rrmse_val, rho_val = calculate_metrics(y_test, y_hat)

    print(f"rRMSE: {rrmse_val:.4f}")
    print(f"Pearson Correlation (rho): {rho_val:.4f}")