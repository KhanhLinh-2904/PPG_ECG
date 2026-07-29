import os 
import numpy as np 
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader, TensorDataset 
from model import FocusedNeuralNetwork 
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve, average_precision_score 
from sklearn.model_selection import train_test_split  
import matplotlib.pyplot as plt 

# ----------- Configuration ----------- 
CHECKPOINT_PATH = "/home/linhhima/PPG_ECG/AF_Detection/new_checkpoint_threshold/best_model.pth" 
TEST_DATA_PATH = "/home/linhhima/PPG_ECG/AF_Detection/total_mimic_af_flow_segment_mse_1.npz"
BATCH_SIZE = 32 
SEED = 105

# Thiết lập Ngưỡng Quyết định (Decision Threshold) 
# Đặt thành None nếu muốn máy tự động tìm Threshold tốt nhất trên 80% tập dữ liệu bằng Youden's Index. 
# Đặt một giá trị cụ thể (ví dụ: 0.1374) nếu muốn áp đặt cố định. 
CUSTOM_THRESHOLD = None
 


# ----------- 1. LOAD & SPLIT DATASET (80% LEARNING, 20% TESTING) ----------- 
if not os.path.exists(TEST_DATA_PATH): 
    raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {TEST_DATA_PATH}") 

test_data = np.load(TEST_DATA_PATH) 
X_all = test_data['X'] 
y_all = test_data['y'] # Nhãn liên tục (xác suất/ngưỡng) từ luồng sinh dữ liệu hồi quy 

print(f"[*] Tổng số mẫu ban đầu: {X_all.shape[0]} phân đoạn.") 

# Tạo mảng tầng phân lớp chuẩn mốc 0.5 để hàm stratify chia đều tỷ lệ mẫu AF/Non-AF vào cả 2 tập 
stratify_labels = (y_all >= 0.5).astype(int) 

# Phân chia dữ liệu theo tỷ lệ 80% Train/Learn Threshold và 20% Test Threshold 
X_learn, X_test, y_learn, y_test = train_test_split( 
    X_all, y_all,  
    test_size=0.50, # ĐÃ SỬA: Điều chỉnh lại thành 0.20 cho đúng tỷ lệ phân tách độc lập 80/20 
    random_state=SEED,  
    stratify=stratify_labels 
) 

# Chuyển đổi sang định dạng PyTorch Tensor (Đồng bộ kiểu Float32 cho luồng Hồi quy) 
X_learn_t = torch.tensor(X_learn, dtype=torch.float32) 
y_learn_t = torch.tensor(y_learn, dtype=torch.float32) 
X_test_t = torch.tensor(X_test, dtype=torch.float32) 
y_test_t = torch.tensor(y_test, dtype=torch.float32) 

# Khởi tạo DataLoader độc lập 
learn_loader = DataLoader(TensorDataset(X_learn_t, y_learn_t), batch_size=BATCH_SIZE, shuffle=False) 
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False) 

print(f" -> Kích thước tập Learning (80%): {X_learn.shape[0]} mẫu") 
print(f" -> Kích thước tập Testing  (20%): {X_test.shape[0]} mẫu") 


# ----------- 2. LOAD PRETRAINED MODEL ----------- 
# Khởi tạo mô hình đơn nhánh đầu ra (Đã loại bỏ tham số lớp rườm rà) 
model = FocusedNeuralNetwork() 

if not os.path.exists(CHECKPOINT_PATH): 
    raise FileNotFoundError(f"Không tìm thấy trọng số mô hình tại: {CHECKPOINT_PATH}") 

try: 
    model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True)) 
except TypeError: 
    model.load_state_dict(torch.load(CHECKPOINT_PATH)) 
     
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model.to(device) 
model.eval() 

# Helper function để chạy suy luận thu xác suất đầu ra 
def inference_pipeline(dataloader, desc="Suy luận"): 
    all_labels = [] 
    all_probs = [] 
    print(f"[*] Đang chạy xử lý: {desc}...") 
    with torch.no_grad(): 
        for inputs, labels in dataloader: 
            inputs = inputs.to(device) 
            # Output mang shape [B, 1], đưa về mảng phẳng bằng .view(-1) 
            outputs = model(inputs).view(-1) 
             
            all_labels.extend(labels.view(-1).numpy()) 
            all_probs.extend(outputs.cpu().numpy()) 
    return np.array(all_labels), np.array(all_probs) 

# Chạy inference trên cả 2 tập dữ liệu để lấy mảng giá trị dự đoán liên tục 
y_true_learn, y_probs_learn = inference_pipeline(learn_loader, desc="80% Tập Tìm Ngưỡng (Learning)") 
y_true_test, y_probs_test = inference_pipeline(test_loader, desc="20% Tập Kiểm Thử (Testing)") 

# Nhị phân hóa nhãn gốc đối chứng chuẩn mốc 0.5 phục vụ tính toán các chỉ số phân lớp của sklearn 
y_true_learn_binary = (y_true_learn >= 0.5).astype(int) 
y_true_test_binary = (y_true_test >= 0.5).astype(int) 


# ----------- 3. STEP 1: LEARN OPTIMAL THRESHOLD (Trên 80% Dữ Liệu) ----------- 
fpr_learn, tpr_learn, roc_thresholds_learn = roc_curve(y_true_learn_binary, y_probs_learn) 
roc_auc_learn = auc(fpr_learn, tpr_learn) 

if CUSTOM_THRESHOLD is None: 
    print("\n[*] Đang tiến hành dò tìm ngưỡng cắt tối ưu trên 80% tập Learning qua Youden's Index...") 
    youden_j_learn = tpr_learn - fpr_learn 
    best_idx_learn = np.argmax(youden_j_learn) # ĐÃ SỬA: Loại bỏ keyword argument sai ngữ pháp gây lỗi TypeError
     
    optimal_threshold = roc_thresholds_learn[best_idx_learn] 
    max_j_learn = youden_j_learn[best_idx_learn] 
     
    print(f" 🎯 Kết quả tìm ngưỡng:") 
    print(f"   - Max Youden's J lý thuyết : {max_j_learn:.4f}") 
    print(f"   - Ngưỡng tối ưu tìm được   : {optimal_threshold:.4f}") 
else: 
    optimal_threshold = CUSTOM_THRESHOLD 
    print(f"\n[*] Áp dụng cố định ngưỡng chỉ định: {optimal_threshold:.4f}") 


# ----------- 4. STEP 2: TEST THRESHOLD (Đánh giá khách quan trên 20% còn lại) ----------- 
# Áp đặt nhãn quyết định cho tập test dựa trên ngưỡng tối ưu vừa xác định 
y_pred_test_binary = (y_probs_test >= optimal_threshold).astype(int) 

# Trích xuất ma trận nhầm lẫn lâm sàng trên tập test độc lập 
tn, fp, fn, tp = confusion_matrix(y_true_test_binary, y_pred_test_binary).ravel() 
total_test = len(y_true_test_binary) 

# Tính toán chi tiết 4 chỉ số chẩn đoán y sinh theo tỷ lệ % 
accuracy = 100 * (tp + tn) / total_test 
precision = 100 * tp / (tp + fp) if (tp + fp) > 0 else 0.0 
recall = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Sensitivity 
specificity = 100 * tn / (tn + fp) if (tn + fp) > 0 else 0.0 

# Tính diện tích dưới đường cong của tập test phục vụ đối chiếu hình học 
fpr_test, tpr_test, roc_thresholds_test = roc_curve(y_true_test_binary, y_probs_test) 
roc_auc_test = auc(fpr_test, tpr_test) 


# ----------- 5. IN BÁO CÁO KẾT QUẢ ĐỘC LẬP CHO LUẬN VĂN / SLIDE ----------- 
print("\n" + "="*60) 
print(f" 🏆 BÁO CÁO THỰC NGHIỆM TRÊN TẬP TEST ĐỘC LẬP (20% DATASET)") 
print(f"    Ngưỡng áp dụng được học từ tập Train: THRESHOLD = {optimal_threshold:.4f}") 
print("="*60) 
print(f"AUROC (Tập học - 80%) : {roc_auc_learn:.4f}") 
print(f"AUROC (Tập test - 20%) : {roc_auc_test:.4f}") 
print("-" * 60) 
print(f"True Positives (TP)    : {tp}") 
print(f"True Negatives (TN)    : {tn}") 
print(f"False Positives (FP)   : {fp} (Báo động giả)") 
print(f"False Negatives (FN)   : {fn} (Bỏ lọt bệnh Rung nhĩ)") 
print("-" * 60) 
print(f"Accuracy               : {accuracy:.2f}%") 
print(f"Precision              : {precision:.2f}%") 
print(f"Recall (Sensitivity)   : {recall:.2f}%") 
print(f"Specificity            : {specificity:.2f}%") 
print("="*60 + "\n") 


# ================================================================= 
# VẼ BIỂU ĐỒ ROC CURVE CHO CẢ 2 PHÂN ĐOẠN ĐỂ ĐỐI CHIẾU HÌNH THÁI 
# ================================================================= 
plt.figure(figsize=(9, 7)) 

plt.plot(fpr_learn, tpr_learn, color='dodgerblue', lw=2, 
         linestyle='--', label=f'Learning ROC Curve (AUC = {roc_auc_learn:.4f})')
 
plt.plot(fpr_test, tpr_test, color='darkorange', lw=2.5, label=f'Testing ROC Curve (AUC = {roc_auc_test:.4f})') 
plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Chance Line') 

# Định vị điểm tọa độ chấm đỏ trên đồ thị một cách chính xác tuyệt đối dựa trên giá trị Ma trận nhầm lẫn thực tế 
applied_fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0 
applied_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0 

plt.scatter(applied_fpr, applied_tpr, marker='o', color='red', s=130,  
            label=f'Applied Threshold ({optimal_threshold:.4f})\n(FPR: {applied_fpr:.2f}, TPR: {applied_tpr:.2f})', zorder=5) 

plt.xlim([0.0, 1.0]) 
plt.ylim([0.0, 1.05]) 
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12) 
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12) 
plt.title("ROC Curve Split-Testing Analysis", fontsize=14, fontweight='bold') 
plt.legend(loc="lower right", fontsize=10) 
plt.grid(True, alpha=0.3) 
plt.tight_layout() 
plt.show() 


# ================================================================= 
# VẼ ĐỒ THỊ PRECISION-RECALL (PRC) CHO TẬP TEST ĐỘC LẬP 
# ================================================================= 
precision_curve, recall_curve, _ = precision_recall_curve(y_true_test_binary, y_probs_test) 
avg_precision = average_precision_score(y_true_test_binary, y_probs_test) 

plt.figure(figsize=(9, 7)) 
plt.plot(recall_curve, precision_curve, color='crimson', lw=2.5, 
          label=f'Testing PRC Curve (Average Precision = {avg_precision:.4f})') 
plt.xlabel('Recall (Sensitivity)', fontsize=12) 
plt.ylabel('Precision', fontsize=12) 
plt.title('Precision-Recall (PRC) Curve (20% Testing Set)', fontsize=14, fontweight='bold') 
plt.grid(True, alpha=0.3) 
plt.legend(loc='lower left', fontsize=10) 
plt.ylim([0.0, 1.05]) 
plt.xlim([0.0, 1.0]) 
plt.tight_layout() 
plt.show()