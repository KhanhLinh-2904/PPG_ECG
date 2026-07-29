import numpy as np
from sklearn.model_selection import train_test_split

def print_class_distribution(y_data, label_name):
    """Hàm hỗ trợ đếm và in ra số lượng phân lớp AF và Non-AF"""
    # Ép kiểu về số nguyên để đếm chính xác
    y_ints = y_data.astype(int)
    
    # Đếm số lượng phần tử của từng lớp
    # Giả định thông thường: 1 là AF, 0 là Non-AF
    af_count = np.sum(y_ints == 1)
    non_af_count = np.sum(y_ints == 0)
    total = len(y_ints)
    
    print(f"📊 Phân phối phân lớp trong tập [{label_name}]:")
    print(f"   - Số lượng segments AF (Nhãn 1):     {af_count:<6} ({af_count/total*100:.2f}%)")
    print(f"   - Số lượng segments Non-AF (Nhãn 0): {non_af_count:<6} ({non_af_count/total*100:.2f}%)")
    print(f"   - Tổng cộng số lượng segments:       {total}")

def split_training_data():
    # 1. Đường dẫn các file dữ liệu
    data_path = '/home/linhhima/PPG_ECG/AF_Detection/detect_af_MIT_BIH_train.npz'
    train_out_path = '/home/linhhima/PPG_ECG/AF_Detection/detect_af_MIT_BIH_new_train.npz'
    val_out_path = '/home/linhhima/PPG_ECG/AF_Detection/detect_af_MIT_BIH_new_val.npz'
    
    print(f"--- Đang tải dữ liệu gốc từ: {data_path} ---")
    # 2. Đọc file dữ liệu nén .npz gốc
    raw_data = np.load(data_path)
    X = raw_data['X']
    y = raw_data['y']
    
    print(f"Kích thước dữ liệu gốc: X = {X.shape}, y = {y.shape}\n")
    
    # 3. Chia tách dữ liệu theo tỷ lệ 3:1 (train_size = 0.75, test_size = 0.25)
    # Thêm tham số stratify=y để đảm bảo tỷ lệ AF/Non-AF ở 2 tập mới giống y hệt tập gốc
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=0.25, 
        random_state=40, 
        shuffle=True,
        stratify=y
    )
    
    # 4. In thông tin chi tiết số lượng phân lớp của tập mới
    print_class_distribution(y_train, "NEW TRAIN")
    print("-" * 50)
    print_class_distribution(y_val, "NEW VALIDATION")
    
    # 5. Lưu dữ liệu đã chia vào các file .npz mới
    print("\n--- Đang tiến hành lưu các tập dữ liệu mới ---")
    
    np.savez(train_out_path, X=X_train, y=y_train)
    print(f"-> Đã lưu tập TRAIN mới thành công tại: {train_out_path}")
    
    np.savez(val_out_path, X=X_val, y=y_val)
    print(f"-> Đã lưu tập VAL mới thành công tại:   {val_out_path}")
    
    print("\n[SUCCESS] Quá trình phân tách dữ liệu và thống kê hoàn tất!")

if __name__ == '__main__':
    split_training_data()