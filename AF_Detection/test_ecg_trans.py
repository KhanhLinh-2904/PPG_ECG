import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# Thư viện tính toán các chỉ số đánh giá chuyên sâu
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from ecg_transform import ECGTransformerModel, ECGAFClassifier

if __name__ == "__main__":
    # 1. Cấu hình thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Bắt đầu quá trình kiểm thử trên thiết bị: {device}")

    # ==============================================================================
    # 2. TẢI DỮ LIỆU KIỂM THỬ (TEST DATA)
    # ==============================================================================
    print("⏳ Đang tải dữ liệu Test...")
    try:
        # Thay bằng file dữ liệu test thực tế của bạn
        test_data = np.load('AF_Detection/total_ecg_reconstructions.npz')
        X_test = torch.tensor(test_data["ecgs"], dtype=torch.float32)
        y_test = torch.tensor(test_data["labels"], dtype=torch.long)
    except FileNotFoundError:
        print("⚠️ Không tìm thấy file dữ liệu, sử dụng dữ liệu giả lập (Dummy Data) để test code.")
        X_test = torch.randn(50, 1, 2400) # 50 mẫu test
        y_test = torch.randint(0, 2, (50,))

    # Bảo vệ định dạng shape [Batch, Channels, Length]
    if X_test.dim() == 2:
        X_test = X_test.unsqueeze(1)

    print(f"Kích thước tập Test: {X_test.shape}")
    
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) # Lúc test không cần shuffle

    # ==============================================================================
    # 3. KHỞI TẠO LẠI KIẾN TRÚC MÔ HÌNH (Phải giống hệt lúc Train)
    # ==============================================================================
    print("🧠 Khởi tạo kiến trúc mô hình...")
    EMBED_DIM = 256
    
    transformer_encoder = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d_model=EMBED_DIM, nhead=8, dim_feedforward=1024, dropout=0.1, batch_first=True), 
        num_layers=4
    )
    
    backbone = ECGTransformerModel(
        in_channels=X_test.shape[1],
        embed_dim=EMBED_DIM,
        conv_layers=[(256, 10, 5), (256, 3, 2), (256, 3, 2)], 
        vq_dim=EMBED_DIM,
        transformer_encoder=transformer_encoder
    )

    model = ECGAFClassifier(
        backbone=backbone,
        embed_dim=EMBED_DIM,
        num_classes=2,
        dropout_prob=0.0, # Lúc test tự động tắt, gán 0 cho chắc chắn
        freeze_backbone=False 
    ).to(device)

    # ==============================================================================
    # 4. TẢI TRỌNG SỐ ĐÃ HUẤN LUYỆN (LOAD CHECKPOINT)
    # ==============================================================================
    MODEL_PATH = "AF_Detection/checkpoints/final_af_classifier.pth"
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"✅ Đã tải thành công trọng số từ: {MODEL_PATH}")
    else:
        print(f"❌ LỖI: Không tìm thấy file trọng số tại {MODEL_PATH}. Vui lòng kiểm tra lại!")
        exit()

    # ==============================================================================
    # 5. VÒNG LẶP KIỂM THỬ (INFERENCE LOOP)
    # ==============================================================================
    print("\n" + "="*50)
    print("🔍 ĐANG TIẾN HÀNH DỰ ĐOÁN...")
    print("="*50)
    
    # Chuyển mô hình sang chế độ Evaluation (CỰC KỲ QUAN TRỌNG)
    # Tắt Dropout, Tắt Batch/Layer Norm learning, Tắt Masking m của xương sống
    model.eval() 
    
    all_preds = []
    all_labels = []

    # Bật context no_grad để tắt tính toán đạo hàm, giúp chạy siêu tốc và tiết kiệm RAM
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            logits = model(inputs)
            
            # Lấy class có xác suất cao nhất (0 hoặc 1)
            _, predictions = torch.max(logits, dim=1)
            
            # Đưa kết quả từ GPU về lại CPU để tính toán bằng sklearn
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # ==============================================================================
    # 6. TÍNH TOÁN VÀ IN CÁC CHỈ SỐ ĐÁNH GIÁ (METRICS)
    # ==============================================================================
    acc = accuracy_score(all_labels, all_preds)
    # Đặt average='binary' và pos_label=1 (giả định 1 là lớp Rung nhĩ - AF)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    print("\n📊 BÁO CÁO KẾT QUẢ KIỂM THỬ (TEST REPORT):")
    print(f"  • Độ chính xác tổng thể (Accuracy) : {acc * 100:.2f}%")
    print(f"  • Độ chuẩn xác (Precision)         : {prec * 100:.2f}%")
    print(f"  • Độ nhạy / Thu hồi (Recall)       : {rec * 100:.2f}%")
    print(f"  • Điểm F1-Score                    : {f1 * 100:.2f}%")
    
    print("\n📉 MA TRẬN NHẦM LẪN (CONFUSION MATRIX):")
    print("                  Dự đoán Non-AF (0) | Dự đoán AF (1)")
    print(f"Thực tế Non-AF (0) |        {cm[0][0]:<10} |      {cm[0][1]:<10}")
    print(f"Thực tế AF (1)     |        {cm[1][0]:<10} |      {cm[1][1]:<10}")