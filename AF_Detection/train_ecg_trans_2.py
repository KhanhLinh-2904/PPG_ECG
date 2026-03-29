import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import random
from sklearn.cluster import MiniBatchKMeans # Thư viện cho K-Means
from ecg_transform_2 import ECGTransformerModel, ECGAFClassifier, HuBERTCrossEntropyLoss

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    # ==========================================
    # ⚙️ CẤU HÌNH CHẠY (BẬT/TẮT CÁC GIAI ĐOẠN)
    # ==========================================
    DO_PRETRAIN = True  
    DO_FINETUNE = True   
    NUM_CLUSTERS = 100 # Số lượng cụm K-Means cho HuBERT
    # ==========================================

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Chạy trên thiết bị: {device}")

    # Tạo thư mục lưu trọng số
    CHECKPOINT_DIR = "AF_Detection/checkpoints_hubert"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    pretrained_path = os.path.join(CHECKPOINT_DIR, "pretrained_hubert_backbone.pth")
    print(f"📁 Thư mục lưu trọng số: {CHECKPOINT_DIR}")

    # --- CHUẨN BỊ DỮ LIỆU TRAIN ---
    print("⏳ Đang tải dữ liệu Train...")
    try:
        train_data = np.load('processed_data/MIT_BIH_train_segments.npz')
        X_train = torch.tensor(train_data["ecgs"], dtype=torch.float32)
        y_train = torch.tensor(train_data["labels"], dtype=torch.long)
    except FileNotFoundError:
        print("⚠️ Không tìm thấy file dữ liệu Train, sử dụng dữ liệu giả lập (Dummy Data).")
        X_train = torch.randn(100, 1, 2400)
        y_train = torch.randint(0, 2, (100,))

    if X_train.dim() == 2:
        X_train = X_train.unsqueeze(1)

    # Loader cho Fine-Tuning (Dùng nhãn thật y_train)
    finetune_dataset = TensorDataset(X_train, y_train)
    finetune_loader = DataLoader(finetune_dataset, batch_size=32, shuffle=True)

    # --- CHUẨN BỊ 3 TẬP DỮ LIỆU TEST ---
    print("⏳ Đang tải các tập dữ liệu Test...")
    test_loaders = {}
    
    test_files = {
        "MIT-BIH": 'processed_data/MIT_BIH_test_segments.npz',
        "Total-Recon": 'AF_Detection/total_ecg_reconstructions.npz',
        "DeepBeat-Recon": 'AF_Detection/deepbeat_ecg_reconstructions.npz'
    }

    for name, path in test_files.items():
        try:
            t_data = np.load(path)
            X_t = torch.tensor(t_data["ecgs"], dtype=torch.float32)
            y_t = torch.tensor(t_data["labels"], dtype=torch.long)
            
            if X_t.dim() == 2: X_t = X_t.unsqueeze(1)
                
            t_dataset = TensorDataset(X_t, y_t)
            test_loaders[name] = DataLoader(t_dataset, batch_size=32, shuffle=False)
            print(f"  ✅ Đã tải thành công tập Test: {name} (Kích thước: {X_t.shape[0]} mẫu)")
        except FileNotFoundError:
            print(f"  ⚠️ Lỗi: Không tìm thấy file '{path}'. Bỏ qua tập này.")

    # --- (BACKBONE) ---
    print("🧠 Khởi tạo mô hình Backbone (HuBERT Style)...")
    EMBED_DIM = 256
    transformer_encoder = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d_model=EMBED_DIM, nhead=8, dim_feedforward=1024, dropout=0.1, batch_first=True), 
        num_layers=4
    )
    
    backbone = ECGTransformerModel(
        in_channels=X_train.shape[1],
        embed_dim=EMBED_DIM,
        num_clusters=NUM_CLUSTERS, # Thay vq_dim bằng num_clusters
        conv_layers=[(256, 10, 5), (256, 3, 2), (256, 3, 2)], 
        feature_grad_mult=1.0,
        transformer_encoder=transformer_encoder
    ).to(device)


    # ==========================================================================
    # GIAI ĐOẠN 1: PRE-TRAINING (HUBERT)
    # ==========================================================================
    if DO_PRETRAIN:
        print("\n" + "="*50)
        print("🌟 GIAI ĐOẠN 1: TẠO NHÃN K-MEANS & PRE-TRAINING")
        print("="*50)
        
        # 1. TẠO NHÃN GIẢ BẰNG K-MEANS
        print("⏳ Bắt đầu trích xuất đặc trưng CNN để chạy K-means...")
        backbone.eval()
        all_features = []
        
        # Dùng DataLoader không shuffle để trích xuất feature giữ đúng thứ tự
        feature_extract_loader = DataLoader(TensorDataset(X_train), batch_size=64, shuffle=False)
        
        with torch.no_grad():
            for (batch_x,) in feature_extract_loader:
                batch_x = batch_x.to(device)
                # Trích xuất đặc trưng từ CNN (không qua Transformer, không mask)
                features = backbone.feature_extractor(batch_x)
                features = backbone.layer_norm(features.transpose(1, 2))
                # Chuyển về CPU và numpy
                all_features.append(features.cpu().numpy())
                
        # Nối lại và gom phẳng: shape (Toàn bộ điểm thời gian, Embed_Dim)
        all_features_np = np.concatenate(all_features, axis=0)
        B, T, D = all_features_np.shape
        all_features_flat = all_features_np.reshape(-1, D)
        
        print(f"⏳ Đang chạy K-means với {NUM_CLUSTERS} cụm trên {all_features_flat.shape[0]} điểm dữ liệu...")
        kmeans = MiniBatchKMeans(n_clusters=NUM_CLUSTERS, batch_size=10000, random_state=42)
        kmeans_labels_flat = kmeans.fit_predict(all_features_flat)
        
        # Định dạng lại thành shape ban đầu (Batch, Time)
        kmeans_labels = kmeans_labels_flat.reshape(B, T)
        kmeans_labels_tensor = torch.tensor(kmeans_labels, dtype=torch.long)
        print("✅ Hoàn tất tạo Pseudo-labels!")

        # 2. TẠO DATALOADER CHO PRE-TRAIN (Ép ECG gốc với nhãn K-Means)
        pretrain_dataset = TensorDataset(X_train, kmeans_labels_tensor)
        pretrain_loader = DataLoader(pretrain_dataset, batch_size=32, shuffle=True)

        # 3. TIẾN HÀNH PRE-TRAIN
        pretrain_epochs = 15
        pretrain_criterion = HuBERTCrossEntropyLoss().to(device) # Dùng hàm loss HuBERT
        pretrain_optimizer = torch.optim.AdamW(backbone.parameters(), lr=5e-4)

        for epoch in range(pretrain_epochs):
            backbone.train()
            running_loss = 0.0
            
            for batch_idx, (inputs, target_kmeans) in enumerate(pretrain_loader):
                inputs, target_kmeans = inputs.to(device), target_kmeans.to(device)
                
                pretrain_optimizer.zero_grad()
                outputs = backbone(inputs)
                
                # Gọi hàm Loss truyền vào: logits, nhãn K-Means, và vị trí mask
                loss = pretrain_criterion(
                    logits=outputs["logits"],
                    target_labels=target_kmeans,
                    mask_indices=outputs["mask_indices"]
                )
                
                if loss.requires_grad:
                    loss.backward()
                    pretrain_optimizer.step()
                    
                running_loss += loss.item()
                
            print(f"Pre-train Epoch [{epoch+1}/{pretrain_epochs}] | HuBERT Loss: {running_loss/len(pretrain_loader):.4f}")

        torch.save(backbone.state_dict(), pretrained_path)
        print(f"✅ Lưu Backbone thành công tại: {pretrained_path}")
    else:
        print("\n" + "="*50)
        print("⏩ BỎ QUA PRE-TRAINING. TẢI TRỌNG SỐ TỪ FILE...")
        print("="*50)
        if os.path.exists(pretrained_path):
            backbone.load_state_dict(torch.load(pretrained_path, map_location=device))
            print(f"✅ Đã tải trọng số Backbone từ: {pretrained_path}")
        else:
            print(f"⚠️ CẢNH BÁO: Không tìm thấy '{pretrained_path}'. Backbone sẽ dùng trọng số khởi tạo ngẫu nhiên!")


    # ==========================================================================
    # GIAI ĐOẠN 2: FINE-TUNING CHO PHÂN LOẠI AF
    # ==========================================================================
    if DO_FINETUNE:
        print("\n" + "="*50)
        print("🎯 GIAI ĐOẠN 2: FINE-TUNING (WARM-UP -> UNFREEZE)")
        print("="*50)

        model = ECGAFClassifier(
            backbone=backbone,
            embed_dim=EMBED_DIM,
            hidden_dim=int(EMBED_DIM/2),
            num_classes=2,
            dropout_prob=0.3,
            freeze_backbone=True
        ).to(device)

        warmup_epochs = 5
        finetune_epochs = 20
        finetune_criterion = nn.CrossEntropyLoss()
        
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        finetune_optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)

        for epoch in range(finetune_epochs):
            
            if epoch == warmup_epochs:
                print("\n" + "🔥"*25)
                print(" HẾT WARM-UP: MỞ ĐÓNG BĂNG BACKBONE ĐỂ FINETUNE TOÀN BỘ")
                print("🔥"*25)
                
                for param in model.backbone.parameters():
                    param.requires_grad = True
                
                finetune_optimizer = torch.optim.AdamW([
                    {'params': model.backbone.parameters(), 'lr': 1e-5},
                    {'params': model.classifier.parameters(), 'lr': 1e-4}
                ])

            model.train()
            if epoch < warmup_epochs:
                model.backbone.eval()

            running_loss = 0.0
            correct_preds = 0
            total_samples = 0
            
            # Sử dụng finetune_loader (chứa nhãn phân loại AF thực sự y_train)
            for batch_idx, (inputs, labels) in enumerate(finetune_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                finetune_optimizer.zero_grad()
                logits = model(inputs)
                loss = finetune_criterion(logits, labels)
                
                loss.backward()
                finetune_optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                _, predictions = torch.max(logits, dim=1)
                correct_preds += torch.sum(predictions == labels).item()
                total_samples += labels.size(0)
                
            epoch_loss = running_loss / total_samples
            epoch_acc = (correct_preds / total_samples) * 100.0

            phase_label = "Warm-up (Head only)" if epoch < warmup_epochs else "Fine-tune (All)"
            print(f"\nEpoch [{epoch+1}/{finetune_epochs}] - {phase_label} | Train Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc:.2f}%")

            if test_loaders:
                model.eval() 
                with torch.no_grad():
                    for test_name, loader in test_loaders.items():
                        test_loss = 0.0
                        test_correct = 0
                        test_total = 0
                        
                        for test_inputs, test_labels in loader:
                            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
                            
                            test_logits = model(test_inputs)
                            t_loss = finetune_criterion(test_logits, test_labels)
                            
                            test_loss += t_loss.item() * test_inputs.size(0)
                            _, test_preds = torch.max(test_logits, dim=1)
                            test_correct += torch.sum(test_preds == test_labels).item()
                            test_total += test_labels.size(0)
                            
                        epoch_test_loss = test_loss / test_total
                        epoch_test_acc = (test_correct / test_total) * 100.0
                        
                        print(f"  👉 Test [{test_name}]: Loss = {epoch_test_loss:.4f} | Acc = {epoch_test_acc:.2f}%")

        final_path = os.path.join(CHECKPOINT_DIR, "final_af_classifier_hubert.pth")
        torch.save(model.state_dict(), final_path)
        print(f"\n🎉 KẾT THÚC CHU TRÌNH! Lưu mô hình tại: {final_path}")