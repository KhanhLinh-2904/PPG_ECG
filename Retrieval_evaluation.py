import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from CLIP import ECGEssembleCLIP
from load_data import LoadData
import matplotlib.pyplot as plt
import numpy as np
# --- THIẾT LẬP HẰNG SỐ ---
OUTPUT_EMBED_DIM = 128 
INPUT_LENGTH = 300
RECALL_K = [1,2,3,4, 5,6,7,8,9, 10]

def calculate_recall_at_k(similarity_matrix, k_values):
    """
    Tính Recall@K cho ma trận tương đồng (dự đoán).
    Giả định: Mẫu đúng nằm trên đường chéo (chỉ số i của truy vấn khớp với chỉ số i của mục tiêu).
    """
    n_samples = similarity_matrix.size(0)
    
    # Sắp xếp các điểm tương đồng theo thứ tự giảm dần
    # indices shape: [N_query, N_targets]
    indices = torch.argsort(similarity_matrix, dim=1, descending=True)
    
    # Tạo ma trận vị trí đúng (ground truth indices)
    # Target indices là [0, 1, 2, ..., N-1]
    ground_truth = torch.arange(n_samples).to(similarity_matrix.device).unsqueeze(1)
    
    recall_results = {}
    
    for k in k_values:
        # Lấy K vị trí hàng đầu
        top_k_indices = indices[:, :k]
        
        # Kiểm tra xem chỉ số đúng có nằm trong K vị trí hàng đầu không
        # is_hit shape: [N_query, K]. is_hit[i, j] = True nếu target đúng nằm ở vị trí j.
        is_hit = (top_k_indices == ground_truth).any(dim=1)
        
        # Tỷ lệ tìm thấy (Recall)
        recall = is_hit.float().sum() / n_samples
        recall_results[k] = recall.item() * 100
        
    return recall_results

# --- HÀM ĐÁNH GIÁ RETRIEVAL ---

def evaluate_retrieval(model, dataloader, device, k_values):
    model.eval()
    
    print("\n--- Mã hóa các vector nhúng (Embedding) ---")
    
    with torch.no_grad():
        # Sửa lỗi: dataloader trả về 3 giá trị (ecg, ppg, label)
        for ecg, ppg, _ in tqdm(dataloader, desc="Encoding Features", unit="batch"): 
            
            # Sửa lỗi: Thêm chiều channel (1) cho ECG và PPG
            ecg = ecg.to(device).float().unsqueeze(1)
            ppg = ppg.to(device).float().unsqueeze(1)
            logits_per_ecg, logits_per_ppg = model(ecg, ppg)

    similarity_matrix = logits_per_ecg
    # 3. Đánh giá ECG-to-PPG Retrieval
    print("--- Đánh giá ECG-to-PPG Retrieval ---")
    # Truy vấn bằng ECG (hàng) và tìm kiếm trong PPG (cột)
    ecg_to_ppg_recall = calculate_recall_at_k(similarity_matrix, k_values)
    
    # 4. Đánh giá PPG-to-ECG Retrieval
    print("--- Đánh giá PPG-to-ECG Retrieval ---")
    # Truy vấn bằng PPG (cột) và tìm kiếm trong ECG (hàng)
    # Tương đương với việc chuyển vị ma trận tương đồng: S.t()
    ppg_to_ecg_recall = calculate_recall_at_k(similarity_matrix.t(), k_values)
    
    return ecg_to_ppg_recall, ppg_to_ecg_recall

def plot_recall_at_k(ecg_to_ppg_recall, ppg_to_ecg_recall, k_values):
    """
    Draws two separate Line Charts comparing the Recall@K results for ECG-to-PPG and PPG-to-ECG Retrieval.
    
    Args:
        ecg_to_ppg_recall (dict): Dictionary containing Recall@K results for ECG-to-PPG.
        ppg_to_ecg_recall (dict): Dictionary containing Recall@K results for PPG-to-ECG.
        k_values (list): List of K values used.
    """
    
    # 1. Prepare data
    recall_ecg_to_ppg = [ecg_to_ppg_recall[k] for k in k_values]
    recall_ppg_to_ecg = [ppg_to_ecg_recall[k] for k in k_values]
    x = np.arange(len(k_values)) 
    
    # X-axis labels for xticks
    x_labels = [f'@{k}' for k in k_values]

    # 2. Create Figure with Two Subplots (1 row, 2 columns)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: ECG to PPG Retrieval (ax[0]) ---
    ax[0].plot(x, recall_ecg_to_ppg, 
            marker='o',           # Circle marker
            linestyle='-',        # Solid line
            color='#E53935',      # Red
            linewidth=2, 
            label='ECG to PPG Retrieval')

    # Set title and labels for ax[0]
    ax[0].set_title('ECG to PPG Retrieval (Recall@K)', fontsize=14, fontweight='bold')
    ax[0].set_ylabel('Recall Percentage (%)', fontsize=12)
    ax[0].set_xlabel('K Value (Top K Results)', fontsize=12)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(x_labels)
    ax[0].set_ylim(0, 105) 
    ax[0].grid(axis='y', linestyle='--', alpha=0.6)
    ax[0].legend(loc='lower right')
    
    # Add data labels for ax[0]
    for i, val in enumerate(recall_ecg_to_ppg):
        ax[0].text(x[i], val + 1.5, f'{val:.2f}%', 
                ha='center', va='bottom', fontsize=9, color='#E53935', fontweight='bold')
    
    
    # --- Plot 2: PPG to ECG Retrieval (ax[1]) ---
    ax[1].plot(x, recall_ppg_to_ecg, 
            marker='s',           # Square marker
            linestyle='-',        # Solid line
            color='#5E35B1',      # Purple
            linewidth=2, 
            label='PPG to ECG Retrieval')

    # Set title and labels for ax[1]
    ax[1].set_title('PPG to ECG Retrieval (Recall@K)', fontsize=14, fontweight='bold')
    ax[1].set_ylabel('Recall Percentage (%)', fontsize=12)
    ax[1].set_xlabel('K Value (Top K Results)', fontsize=12)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(x_labels)
    ax[1].set_ylim(0, 105) 
    ax[1].grid(axis='y', linestyle='--', alpha=0.6)
    ax[1].legend(loc='lower right')
    
    # Add data labels for ax[1]
    for i, val in enumerate(recall_ppg_to_ecg):
        # Place label slightly above the point
        ax[1].text(x[i], val + 1.5, f'{val:.2f}%', 
                ha='center', va='bottom', fontsize=9, color='#5E35B1', fontweight='bold')
        

    # 5. Display the chart
    # Add a main title for the entire figure
    fig.suptitle('Cross-Modal Retrieval Performance Comparison', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle
    plt.show()

# --- KHỐI CHẠY CHÍNH ---

if __name__ == '__main__':
    
    BATCH_SIZE = 64
    
    # 1. Thiết bị (Device Setup)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    # 2. Tải mô hình CLIP đã huấn luyện
    clip_model = ECGEssembleCLIP(embed_dim=OUTPUT_EMBED_DIM).to(device)
    
    try:
        # Giả định file trọng số tồn tại
        clip_model.load_state_dict(torch.load('ecg_ppg_clip_best_model.pth', map_location=device))
        print("\n=> Tải trọng số mô hình CLIP từ 'ecg_ppg_clip_best_model.pth' thành công.")
    except FileNotFoundError:
        print("\n=> Lỗi: Không tìm thấy 'ecg_ppg_clip_best_model.pth'. Chạy đánh giá Retrieval với mô hình chưa huấn luyện (kết quả sẽ ngẫu nhiên/thấp).")
    
    # 3. Tải Dữ liệu Kiểm tra Retrieval (Sử dụng đường dẫn file và class LoadData mới)
    test_retrieval_dataset = LoadData('/home/linhhima/Pre_processing_data/datasets/MIMIC_new_test.npz')
    test_retrieval_loader = DataLoader(test_retrieval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 4. CHẠY ĐÁNH GIÁ RETRIEVAL
    ecg_to_ppg_recall, ppg_to_ecg_recall = evaluate_retrieval(
        clip_model, test_retrieval_loader, device, RECALL_K
    )
    
    # 5. HIỂN THỊ KẾT QUẢ
    
    print(f"\n========================================================")
    print(f"       KẾT QUẢ ĐÁNH GIÁ KHẢ NĂNG KHÔI PHỤC (RETRIEVAL)")
    print(f"========================================================")
    
    print("\nA. ECG-to-PPG Retrieval:")
    for k in RECALL_K:
        print(f"  Recall@{k}: {ecg_to_ppg_recall[k]:.2f}%")
        
    print("\nB. PPG-to-ECG Retrieval:")
    for k in RECALL_K:
        print(f"  Recall@{k}: {ppg_to_ecg_recall[k]:.2f}%")
    print(f"========================================================")
    plot_recall_at_k(ecg_to_ppg_recall, ppg_to_ecg_recall, RECALL_K)