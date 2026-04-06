import numpy as np
from scipy.linalg import fractional_matrix_power

def get_coral_stats(features):
    mu = np.mean(features, axis=0)
    cov = np.cov(features, rowvar=False) + np.eye(features.shape[1]) * 1e-5
    return mu, cov

def apply_coral(source_mu, source_cov, target_features):
    target_mu, target_cov = get_coral_stats(target_features)
    
    cov_s_half = fractional_matrix_power(source_cov, 0.5)
    cov_t_inv_half = fractional_matrix_power(target_cov, -0.5)
    
    target_aligned = np.dot(target_features - target_mu, cov_t_inv_half)
    target_aligned = np.dot(target_aligned, cov_s_half) + source_mu
    
    return target_aligned.real