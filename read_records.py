import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

def clean_val(val):
    """Giải mã bytes và xử lý mảng số."""
    if isinstance(val, bytes):
        return val.decode('utf-8')
    return val

def print_all_info(obj, indent=0):
    """Hàm đệ quy để in mọi ngóc ngách của struct."""
    spacing = "  " * indent
    if isinstance(obj, sio.matlab.mat_struct):
        for field in obj._fieldnames:
            child = getattr(obj, field)
            if isinstance(child, (sio.matlab.mat_struct, np.ndarray)) and not isinstance(child, (int, float, str, bytes)):
                print(f"{spacing}id: {field} ->")
                print_all_info(child, indent + 1)
            else:
                print(f"{spacing}{field}: {clean_val(child)}")
    elif isinstance(obj, np.ndarray) and obj.dtype == object:
        for i, item in enumerate(obj):
            print(f"{spacing}[Index {i}] ->")
            print_all_info(item, indent + 1)

# 1. Tải file
file_path = 'Records.mat'
data = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)
records = data['records']


for idx in range(len(records)):
    print(f"{'='*30} THÔNG TIN CHI TIẾT RECORD {idx} {'='*30}")
    print_all_info(records[idx])

