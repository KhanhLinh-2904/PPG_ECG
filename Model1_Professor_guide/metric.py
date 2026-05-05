import numpy as np
from numba import njit
def calculate_metrics(y_true, y_pred):
 
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # 1. rRMSE (Relative Root Mean-Squared Error)
    #  ||y_true - y_pred||2 / ||y_true||2
    error_vector = y_true - y_pred
    rRMSE = np.linalg.norm(error_vector, ord=2) / np.linalg.norm(y_true, ord=2)
    # 2. Pearson’s Correlation Coefficient (rho)
    #  (y_true - mean_true).T @ (y_pred - mean_pred) / (norm_diff_true * norm_diff_pred)
    y_true_centered = y_true - np.mean(y_true)
    y_pred_centered = y_pred - np.mean(y_pred)

    numerator = np.dot(y_true_centered, y_pred_centered)
    denominator = np.linalg.norm(y_true_centered, ord=2) * np.linalg.norm(y_pred_centered, ord=2)

    rho = numerator / denominator
    
    return rRMSE, rho

@njit
def calculate_dtw_distance(y_true, y_pred, window=100):

    y_t = np.asarray(y_true).reshape(-1)
    y_p = np.asarray(y_pred).reshape(-1)
    
    n = len(y_t)
    m = len(y_p)
    
    w = max(window, abs(n - m))
    
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0.0

    for i in range(1, n + 1):
        start = max(1, i - w)
        end = min(m + 1, i + w + 1)
        
        for j in range(start, end):
            cost = abs(y_t[i-1] - y_p[j-1])
            
            last_min = min(dtw_matrix[i-1, j],   
                           dtw_matrix[i, j-1],    
                           dtw_matrix[i-1, j-1])  
            
            dtw_matrix[i, j] = cost + last_min

    return dtw_matrix[n, m]

def calculate_cosine_similarity(y_true, y_pred):
   
    y_t = np.asanyarray(y_true).flatten()
    y_p = np.asanyarray(y_pred).flatten()
    
    dot_product = np.dot(y_t, y_p)
    
    norm_t = np.linalg.norm(y_t)
    norm_p = np.linalg.norm(y_p)
    
    if norm_t == 0 or norm_p == 0:
        return 0.0
        
    return dot_product / (norm_t * norm_p)

if __name__ == "__main__":
    y_test = [0.1, 0.2, 0.5, 1.2, 0.5, 0.2, 0.1]
    y_hat  = [0.12, 0.18, 0.48, 1.15, 0.52, 0.21, 0.09]

    rrmse_val, rho_val = calculate_metrics(y_test, y_hat)

    print(f"rRMSE: {rrmse_val:.4f}")
    print(f"Pearson Correlation (rho): {rho_val:.4f}")