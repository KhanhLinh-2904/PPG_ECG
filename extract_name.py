import re

def extract_subject_names(input_file, output_file):
    names = []
    
    # Mở file đọc với encoding utf-8 để tránh lỗi ký tự đặc biệt
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Kiểm tra xem dòng đó có chứa từ khóa 'name:' không
            if 'name:' in line:
                # Sử dụng split để lấy phần nội dung sau dấu ':'
                # Sau đó dùng strip() để xóa khoảng trắng dư thừa
                full_name = line.split('name:')[1].strip()
                
                # Lấy phần đầu trước dấu '-' đầu tiên
                # Ví dụ: p000773-2109... -> p000773
                short_name = full_name.split('-')[0]
                
                # Chỉ thêm vào danh sách nếu chưa có (để tránh trùng lặp nếu record bị lặp)
                if short_name not in names:
                    names.append(short_name)

    # Lưu kết quả vào file mới
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for name in names:
            f_out.write(name + '\n')
            
    print(f"Đã trích xuất thành công {len(names)} ID và lưu vào {output_file}")

# Chạy hàm
extract_subject_names('text.txt', 'output_names.txt')