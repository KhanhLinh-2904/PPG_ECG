import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import random
from ecg_transform_copy import ECGTransformerModel, ECGAFClassifier, StandardInfoNCEContrastiveLoss

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
    # ==========================================

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Chạy trên thiết bị: {device}")

    # Tạo thư mục lưu trọng số
    CHECKPOINT_DIR = "AF_Detection/checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    pretrained_path = os.path.join(CHECKPOINT_DIR, "pretrained_backbone.pth")
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

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # --- CHUẨN BỊ 3 TẬP DỮ LIỆU TEST ---
    print("⏳ Đang tải các tập dữ liệu Test...")
    test_loaders = {} # Dùng dictionary để lưu tên dataset và loader tương ứng
    
    test_files = {
        "MIT-BIH": 'processed_data/MIT_BIH_test_segments.npz',
        "Total-Recon": 'AF_Detection/total_ecg_reconstructions.npz',
        "DeepBeat-Recon": 'AF_Detection/deepbeat_ecg_reconstructions.npz'
    }

    for name, path in test_files.items():
        try:
            t_data = np.load(path)
            # Lưu ý: Cần kiểm tra tên khóa (key) trong các file .npz mới có giống với MIT-BIH không.
            # Giả định chúng đều dùng khóa 'ecgs' và 'labels'.
            X_t = torch.tensor(t_data["ecgs"], dtype=torch.float32)
            y_t = torch.tensor(t_data["labels"], dtype=torch.long)
            
            if X_t.dim() == 2:
                X_t = X_t.unsqueeze(1)
                
            t_dataset = TensorDataset(X_t, y_t)
            test_loaders[name] = DataLoader(t_dataset, batch_size=32, shuffle=False)
            print(f"  ✅ Đã tải thành công tập Test: {name} (Kích thước: {X_t.shape[0]} mẫu)")
        except FileNotFoundError:
            print(f"  ⚠️ Lỗi: Không tìm thấy file Test '{path}'. Bỏ qua tập này.")
        except KeyError as e:
            print(f"  ⚠️ Lỗi: Không tìm thấy khóa {e} trong file '{path}'. Vui lòng kiểm tra lại cấu trúc file .npz.")

    if not test_loaders:
        print("⚠️ Không tải được tập Test nào!")

    # --- (BACKBONE) ---
    print("🧠 Khởi tạo mô hình Backbone...")
    EMBED_DIM = 256
    transformer_encoder = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d_model=EMBED_DIM, nhead=8, dim_feedforward=1024, dropout=0.1, batch_first=True), 
        num_layers=4
    )
    
    backbone = ECGTransformerModel(
        in_channels=X_train.shape[1],
        embed_dim=EMBED_DIM,
        conv_layers=[(256, 10, 5), (256, 3, 2), (256, 3, 2)], 
        vq_dim=EMBED_DIM,
        feature_grad_mult = 1.0,
        transformer_encoder=transformer_encoder
    ).to(device)


    # ==========================================================================
    # GIAI ĐOẠN 1: PRE-TRAINING
    # ==========================================================================
    if DO_PRETRAIN:
        print("\n" + "="*50)
        print("🌟 GIAI ĐOẠN 1: PRE-TRAINING BACKBONE")
        print("="*50)
        
        pretrain_epochs = 15
        pretrain_criterion = StandardInfoNCEContrastiveLoss(temperature=0.1).to(device)
        pretrain_optimizer = torch.optim.AdamW(backbone.parameters(), lr=5e-4)

        for epoch in range(pretrain_epochs):
            backbone.train()
            running_loss = 0.0
            
           
            for batch_idx, (inputs, _) in enumerate(train_loader):
                inputs = inputs.to(device)
                
                pretrain_optimizer.zero_grad()
                outputs = backbone(inputs)
                loss, c_loss, vq_loss = pretrain_criterion(
                    local_reps=outputs["local_reps"],
                    q_targets=outputs["q_targets"],
                    mask_indices=outputs["mask_indices"],
                    commitment_loss=outputs["commitment_loss"]
                )
                
                if loss.requires_grad:
                    loss.backward()
                    pretrain_optimizer.step()
                    
                running_loss += loss.item()
                
            print(f"Pre-train Epoch [{epoch+1}/{pretrain_epochs}] | Total Loss: {running_loss/len(train_loader):.4f}")

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
    # GIAI ĐOẠN 2: FINE-TUNING (KẾT HỢP WARM-UP & GRADUAL UNFREEZING)
    # ==========================================================================
    if DO_FINETUNE:
        print("\n" + "="*50)
        print("🎯 GIAI ĐOẠN 2: FINE-TUNING CHO PHÂN LOẠI AF")
        print("="*50)

        # Khởi tạo mô hình với trạng thái ĐÓNG BĂNG backbone ban đầu
        model = ECGAFClassifier(
            backbone=backbone,
            embed_dim=EMBED_DIM,
            hidden_dim=int(EMBED_DIM/2),
            num_classes=2,
            dropout_prob=0.3,
            freeze_backbone=True  # Quan trọng: True để đóng băng lúc đầu
        ).to(device)

        warmup_epochs = 5  # Số epoch chỉ train phần classifier head
        finetune_epochs = 20
        finetune_criterion = nn.CrossEntropyLoss()
        
        # Optimizer ban đầu chỉ huấn luyện phần Classifier với LR lớn
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        finetune_optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)

        for epoch in range(finetune_epochs):
            
            # ------------------------------------------------------------------
            # KIỂM TRA MỐC UNFREEZE
            # ------------------------------------------------------------------
            if epoch == warmup_epochs:
                print("\n" + "🔥"*25)
                print(" HẾT WARM-UP: MỞ ĐÓNG BĂNG BACKBONE ĐỂ FINETUNE TOÀN BỘ")
                print("🔥"*25)
                
                # Mở khóa gradient cho backbone
                for param in model.backbone.parameters():
                    param.requires_grad = True
                
                # Tạo lại Optimizer: 
                # Cấp LR nhỏ (1e-5) cho backbone để không làm mất feature đã học
                # Giữ nguyên LR vừa phải (1e-4) cho phần Classifier
                finetune_optimizer = torch.optim.AdamW([
                    {'params': model.backbone.parameters(), 'lr': 1e-5},
                    {'params': model.classifier.parameters(), 'lr': 1e-4}
                ])
            # ------------------------------------------------------------------

            # Xử lý chế độ Train/Eval riêng rẽ khi đang trong giai đoạn Warm-up
            model.train()
            if epoch < warmup_epochs:
                model.backbone.eval() # Giữ BatchNorm/Dropout của backbone đứng im trong lúc warm-up

            running_loss = 0.0
            correct_preds = 0
            total_samples = 0
            
            for batch_idx, (inputs, labels) in enumerate(train_loader):
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

            # --- ĐÁNH GIÁ TRÊN 3 TẬP TEST ---
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

        final_path = os.path.join(CHECKPOINT_DIR, "final_af_classifier.pth")
        torch.save(model.state_dict(), final_path)
        print(f"\n🎉 KẾT THÚC CHU TRÌNH! Lưu mô hình tại: {final_path}")