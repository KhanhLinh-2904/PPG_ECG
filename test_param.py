import torch

# 1. Đường dẫn tới file weight của bạn
freeze_encoder_ecg = "multitask_best_model_ecg.pth"

# 2. Load file (Dùng map_location='cpu' để an toàn, không tốn VRAM GPU)
print(f"Đang đọc file: {freeze_encoder_ecg}...\n")
checkpoint = torch.load(freeze_encoder_ecg, map_location='cpu')

# 3. Kiểm tra xem file lưu dưới dạng Checkpoint tổng hay chỉ là State Dict
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    print("📌 Đây là file Checkpoint (chứa cả model, optimizer, epoch...).")
    state_dict = checkpoint['model_state_dict']
else:
    print("📌 Đây là file State Dict thuần túy (chỉ chứa trọng số mô hình).")
    state_dict = checkpoint

# 4. In ra toàn bộ tên biến (Key) và kích thước (Shape) của chúng
print("\n" + "="*80)
print(f"{'TÊN LAYER (KEY)':<55} | {'KÍCH THƯỚC (SHAPE)'}")
print("="*80)

total_params = 0
for key, tensor in state_dict.items():
    # tensor.shape trả về một tuple, chuyển sang list cho dễ nhìn
    shape = list(tensor.shape) 
    print(f"{key:<55} | {shape}")
    
    # Tính nhẩm tổng số lượng tham số luôn cho vui
    total_params += tensor.numel()

print("="*80)
print(f"Tổng số lượng tham số (Parameters) trong file: {total_params:,}")