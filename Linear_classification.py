import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
# Giả định các module sau đã được định nghĩa trong các file tương ứng
from CLIP import ECGEssembleCLIP
from load_data import LoadData
from load_data_ecg import LoadDataECG
from load_data_ppg import LoadDataPPG

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# --- CÁC HẰNG SỐ ---
OUTPUT_EMBED_DIM = 128 
INPUT_LENGTH = 2400 
NUM_CLASSES = 2
PROJECTION_HIDDEN_DIM = OUTPUT_EMBED_DIM * 2 

class LinearProbe(nn.Module):
    """
    Mô hình kết hợp ECG Encoder đã đóng băng, một lớp Projection (non-linear) 
    và một lớp phân loại tuyến tính mới.
    """
    def __init__(self, ecg_encoder,ppg_encoder, embed_dim, num_classes):
        super(LinearProbe, self).__init__()
        
        # 1. Đóng băng ECG Encoder (Lấy từ mô hình CLIP)
        self.ecg_encoder = ecg_encoder
        self.ppg_encoder = ppg_encoder

        for param in self.ecg_encoder.parameters():
            param.requires_grad = False
        for param in self.ppg_encoder.parameters():
            param.requires_grad = False
        self.ecg_encoder.eval() # Chắc chắn rằng nó ở chế độ đánh giá
        self.ppg_encoder.eval() # Chắc chắn rằng nó ở chế độ đánh giá

        # 2. Lớp Projection Head (Được huấn luyện)
        # Thực hiện ánh xạ phi tuyến tính: Embed_dim -> Hidden -> Embed_dim
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, PROJECTION_HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Linear(PROJECTION_HIDDEN_DIM, embed_dim)
        )

        # 3. Lớp Phân loại Cuối cùng (Được huấn luyện)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, ecg, ppg):
        # Lấy vector nhúng từ encoder (KHÔNG TÍNH GRADIENT)
        with torch.no_grad():
            if ecg is not None:
                ecg_features = self.ecg_encoder(ecg)
            if ppg is not None:
                ppg_features = self.ppg_encoder(ppg)
            
        # Ứng dụng Projection Head (TÍNH GRADIENT VÀ HUẤN LUYỆN)
        if ecg is not None:
            projected_ecg_features = self.projection_head(ecg_features)
        if ppg is not None:
            projected_ppg_features = self.projection_head(ppg_features)
        
        # Phân loại bằng lớp tuyến tính mới (TÍNH GRADIENT VÀ HUẤN LUYỆN)
        if ecg is not None:
            logits_ecg = self.classifier(projected_ecg_features)
        if ppg is not None:
            logits_ppg = self.classifier(projected_ppg_features)
        if ecg is not None and ppg is not None:
            return logits_ecg, logits_ppg
        elif ecg is not None:
            return logits_ecg
        elif ppg is not None:
            return logits_ppg
# --- HÀM HUẤN LUYỆN LINEAR PROBE (Cập nhật để trả về Loss và Accuracy) ---

def train_linear_probe(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    progress_bar = tqdm(dataloader, desc="Training Linear Probe", unit="batch")
    
    for ecg, ppg, labels in progress_bar:
        ecg = ecg.to(device).float().unsqueeze(1)
        ppg = ppg.to(device).float().unsqueeze(1)

        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        logits_ecg = model(ecg=None, ppg=ppg)
        # logits = (logits_ecg + logits_ppg) / 2
        logits = logits_ecg
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Tính Accuracy cho mục đích tracking/plotting
        _, predicted = torch.max(logits, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
    avg_loss = total_loss / len(dataloader)
    accuracy = (correct_predictions / total_samples) * 100
    return avg_loss, accuracy

# --- HÀM ĐÁNH GIÁ (Validation/Test - Trả về Loss và Accuracy) ---
def evaluate_linear_probe(model, dataloader, criterion, device):
    model.eval()
    correct_predictions = 0
    total_samples = 0
    total_loss = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for ppg, labels in dataloader:
            # Chỉ cần ECG vì đây là Linear Probe trên Encoder ECG
            # ecg = ecg.to(device).float().unsqueeze(1)
            ppg = ppg.to(device).float().unsqueeze(1)

            labels = labels.to(device)
            # logits_ecg = model(ecg, ppg=None)
            # logits_ecg = model(ecg, ppg=None)
            logits_ppg = model(ecg=None, ppg=ppg)


            # logits = (logits_ecg + logits_ppg) / 2
            # logits = logits_ecg
            logits = logits_ppg
            # Tính Loss
            loss = criterion(logits, labels)
            

            # total_loss += loss.item()
            # total_loss += loss.item()
            total_loss += loss.item()
            
            # Tính Accuracy
            _, predicted= torch.max(logits, 1)

            # Thu thập nhãn (chuyển về CPU)
            ##################
            all_labels.append(labels.cpu())
            all_predictions.append(predicted.cpu())

            # ##############
            # total_samples += labels.size(0)
            # correct_predictions += (predicted == labels).sum().item() 
    # Nối tất cả labels và predictions thành tensor 1D
    all_labels_tensor = torch.cat(all_labels)
    all_predictions_tensor = torch.cat(all_predictions)
    
    # Chuyển đổi sang numpy để sử dụng các hàm metrics của scikit-learn
    # LƯU Ý: Đã thêm .cpu() để tránh lỗi nếu các tensor cuối cùng vẫn trên GPU
    y_true = all_labels_tensor.cpu().numpy()
    y_pred = all_predictions_tensor.cpu().numpy() 
    # --- TÍNH TOÁN CÁC METRICS ---
    
    avg_loss = total_loss / len(dataloader)
    
    # Accuracy (Độ chính xác)
    accuracy = accuracy_score(y_true, y_pred) * 100
    
    # Precision, Recall, F1-Score (Giả định Lớp 1 là lớp dương - AF)
    try:
        precision = precision_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
        recall = recall_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
        f1 = f1_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
    except Exception as e:
        print(f"Cảnh báo lỗi khi tính P/R/F1: {e}")
        precision, recall, f1 = 0.0, 0.0, 0.0
        
    # In ra kết quả chi tiết
    print(f"\n--- KẾT QUẢ ĐÁNH GIÁ LINEAR PROBE ---")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    return avg_loss, accuracy, precision, recall, f1   
############################    
    # avg_loss = total_loss / len(dataloader)
    # accuracy = (correct_predictions / total_samples) * 100
    # return avg_loss, accuracy

# --- HÀM VẼ BIỂU ĐỒ ---
def plot_metrics(train_losses, val_losses, train_accs, val_accs):
    epochs = range(1, len(train_losses) + 1)
    
    # 1. Plot Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss', marker='.', linestyle='-')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='.', linestyle='-')
    plt.title('Loss over Epochs', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 2. Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label='Training Accuracy', marker='.', linestyle='-')
    plt.plot(epochs, val_accs, label='Validation Accuracy', marker='.', linestyle='-')
    plt.title('Accuracy over Epochs', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('linear_probe_metrics.png')
    print("\n=> Đã lưu biểu đồ Loss và Accuracy vào 'linear_probe_metrics.png'")
    # plt.show() # Bỏ comment nếu muốn hiển thị trực tiếp trong môi trường có giao diện

# --- KHỐI CHẠY CHÍNH ---

if __name__ == '__main__':
    
    # --- CÁC SIÊU THAM SỐ CHO LINEAR PROBE ---
    LP_EPOCHS = 200 # Giảm số epoch để demo nhanh hơn
    LP_LR = 1e-3
    LP_BATCH_SIZE = 1024
    BEST_MODEL_FILENAME = 'linear_probe_best_model.pth'

    # 1. Thiết bị (Device Setup)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    # 2. Tải mô hình CLIP đã huấn luyện
    clip_model = ECGEssembleCLIP(embed_dim=OUTPUT_EMBED_DIM).to(device)
    
    try:
        # Load trọng số CLIP đã huấn luyện trước đó
        clip_model.load_state_dict(torch.load('ecg_ppg_clip_best_model.pth', map_location=device))
        print("\n=> Tải trọng số mô hình CLIP từ 'ecg_ppg_clip_best_model.pth' thành công.")
    except FileNotFoundError:
        print("\n=> Lỗi: Không tìm thấy 'ecg_ppg_clip_best_model.pth'. Sẽ sử dụng các tham số khởi tạo ngẫu nhiên.")
    
    # Lấy ECG Encoder đã được tải trọng số
    ecg_encoder = clip_model.encode_ecg_resnet
    ppg_encoder = clip_model.encode_ecg_transformer

    # 3. Khởi tạo Linear Probe Model (Non-Linear Probe)
    probe_model = LinearProbe(
        ecg_encoder=ecg_encoder, 
        ppg_encoder=ppg_encoder,
        embed_dim=OUTPUT_EMBED_DIM, 
        num_classes=NUM_CLASSES
    ).to(device)
    
    # CHỈ HUẤN LUYỆN Projection Head VÀ Classifier
    trainable_params = list(probe_model.projection_head.parameters()) + list(probe_model.classifier.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=LP_LR)
    criterion = nn.CrossEntropyLoss()
    
    # 4. Tải Dữ liệu cho Linear Probe
    # Giả định LoadData đã được định nghĩa và hoạt động
    try:
        train_probe_dataset = LoadData('datasets/MIMIC_train.npz')
        val_probe_dataset = LoadData('datasets/MIMIC_val.npz')
        # test_probe_dataset = LoadData('datasets/MIMIC_test.npz')
        test_probe_dataset = LoadDataPPG('datasets/deepbeat_data_extract/test_cleaned_signals.npz')
        # test_probe_dataset = LoadDataECG('datasets/ECG_MIT-BIH_data.npz')


    except Exception as e:
        print(f"\nLỗi khi tải dữ liệu: {e}. Vui lòng kiểm tra file 'load_data.py' và đường dẫn.")
        exit()
    
    train_loader = DataLoader(train_probe_dataset, batch_size=LP_BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_probe_dataset, batch_size=LP_BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_probe_dataset, batch_size=LP_BATCH_SIZE, shuffle=False, num_workers=0)
    
    # # 5. Vòng lặp Huấn luyện Linear Probe (Tích hợp Early Stopping)
    # print("\n--- Bắt đầu Huấn luyện Non-Linear Probe ---")
    
    # best_val_loss = float('inf')

    # # Lists to track metrics
    # train_losses_history = []
    # val_losses_history = []
    # train_accs_history = []
    # val_accs_history = []

    # for epoch in range(1, LP_EPOCHS + 1):
    #     # 1. Training
    #     train_loss, train_accuracy = train_linear_probe(probe_model, train_loader, optimizer, criterion, device)
        
    #     # 2. Validation
    #     val_loss, val_accuracy = evaluate_linear_probe(probe_model, val_loader, criterion, device)
        
    #     # 3. Lưu lịch sử
    #     train_losses_history.append(train_loss)
    #     val_losses_history.append(val_loss)
    #     train_accs_history.append(train_accuracy)
    #     val_accs_history.append(val_accuracy)

    #     print(f"Epoch {epoch}/{LP_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

    #     # 4. Kiểm tra và Lưu mô hình tốt nhất
    #     if val_loss < best_val_loss:
    #         best_val_loss = val_loss
    #         # Lưu lại trạng thái của mô hình (chỉ các tham số có thể huấn luyện)
    #         torch.save(probe_model.state_dict(), BEST_MODEL_FILENAME)
    #         print(f"*** Lưu mô hình tốt nhất với Val Loss: {best_val_loss:.4f} ***")

    # 6. Tải mô hình tốt nhất và Đánh giá trên Test Set
    print("\n--- Tải mô hình tốt nhất để Đánh giá trên Test Set ---")
    
    try:
        # Tải lại trọng số của mô hình có Val Loss thấp nhất
        probe_model.load_state_dict(torch.load(BEST_MODEL_FILENAME, map_location=device))
        # print(f"Tải {BEST_MODEL_FILENAME} thành công. Best Val Loss: {best_val_loss:.4f}")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file mô hình tốt nhất {BEST_MODEL_FILENAME}. Đánh giá bằng mô hình cuối cùng.")
    
    # Đánh giá trên tập Test (sử dụng mô hình tốt nhất đã được tải)
    final_test_loss, final_accuracy, _, _, _ = evaluate_linear_probe(probe_model, test_loader, criterion, device)

    # # 7. Vẽ biểu đồ
    # plot_metrics(train_losses_history, val_losses_history, train_accs_history, val_accs_history)
    
    # print(f"\n========================================================")
    # print(f" KẾT QUẢ ĐÁNH GIÁ CUỐI CÙNG (Dựa trên mô hình tốt nhất)")
    # print(f" Test Loss: {final_test_loss:.4f}")
    # print(f" Final Test Accuracy: {final_accuracy:.2f}%")
    # print(f"========================================================")
