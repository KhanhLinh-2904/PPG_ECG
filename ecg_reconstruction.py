import torch
import torch.nn.functional as F
import pywt
import numpy as np

class SignalReconstructor:
    def __init__(self, dwt_wavelet='db4', dwt_level=3, original_length=2400):
        # SỬA: Tạo đối tượng Wavelet ngay khi khởi tạo
        self.dwt_wavelet = dwt_wavelet
        self.wavelet_obj = pywt.Wavelet(dwt_wavelet) 
        self.dwt_level = dwt_level
        self.original_length = original_length

    def inverse_ecg_dwt(self, features_batch):
        """
        Reconstructs ECG signals from resized DWT features.
        Args:
            features_batch: Tensor [Batch, 4, 306]
        """
        device = features_batch.device
        batch_size = features_batch.shape[0]
        
        # 1. Tính toán độ dài chuẩn (Target Lengths)
        # SỬA: Truyền self.wavelet_obj thay vì chuỗi tên 'db4'
        l1_len = pywt.dwt_coeff_len(self.original_length, self.wavelet_obj, mode='symmetric')
        l2_len = pywt.dwt_coeff_len(l1_len, self.wavelet_obj, mode='symmetric')
        l3_len = pywt.dwt_coeff_len(l2_len, self.wavelet_obj, mode='symmetric')
        
        # 2. Tách và Resize ngược lại (Upsample)
        # Kênh 0 & 1 (Level 3: cA3, cD3) -> Resize về l3_len
        c0_tensor = features_batch[:, 0:1, :] 
        c0_resized = F.interpolate(c0_tensor, size=l3_len, mode='linear', align_corners=False)
        c0_np = c0_resized.squeeze(1).cpu().detach().numpy()
        
        c1_tensor = features_batch[:, 1:2, :] 
        c1_resized = F.interpolate(c1_tensor, size=l3_len, mode='linear', align_corners=False)
        c1_np = c1_resized.squeeze(1).cpu().detach().numpy()
        
        # Kênh 2 (Level 2: cD2) -> Resize về l2_len
        c2_tensor = features_batch[:, 2:3, :] 
        c2_resized = F.interpolate(c2_tensor, size=l2_len, mode='linear', align_corners=False)
        c2_np = c2_resized.squeeze(1).cpu().detach().numpy()
        
        # Kênh 3 (Level 1: cD1) -> Resize về l1_len
        c3_tensor = features_batch[:, 3:4, :] 
        c3_resized = F.interpolate(c3_tensor, size=l1_len, mode='linear', align_corners=False)
        c3_np = c3_resized.squeeze(1).cpu().detach().numpy()
        
        # 3. Dùng PyWavelets để tái tạo (Inverse DWT)
        reconstructed_list = []
        
        for i in range(batch_size):
            # Gom lại thành list: [cA3, cD3, cD2, cD1]
            coeffs = [c0_np[i], c1_np[i], c2_np[i], c3_np[i]]
            
            # Gọi hàm waverec
            # Ở đây vẫn truyền tên chuỗi 'db4' hoặc object đều được, waverec thông minh hơn dwt_coeff_len
            rec_signal = pywt.waverec(coeffs, self.dwt_wavelet, mode='symmetric')
            
            # Fix độ dài
            if len(rec_signal) > self.original_length:
                rec_signal = rec_signal[:self.original_length]
            elif len(rec_signal) < self.original_length:
                rec_signal = np.pad(rec_signal, (0, self.original_length - len(rec_signal)))
                
            reconstructed_list.append(rec_signal)
            
        return torch.tensor(np.array(reconstructed_list)).float().to(device)
    

if __name__ == "__main__":
    # Giả lập dữ liệu features có shape [49, 4, 306]
    dummy_features = torch.randn(49, 4, 306)
    
    reconstructor = SignalReconstructor(dwt_wavelet='db4', dwt_level=3, original_length=2400)
    
    # Thực hiện tái tạo
    ecg_reconstructed = reconstructor.inverse_ecg_dwt(dummy_features)
    
    print("Input Features Shape:", dummy_features.shape)
    print("Reconstructed ECG Shape:", ecg_reconstructed.shape)
    
    # Kiểm tra xem có đúng shape [49, 2400] không
    assert ecg_reconstructed.shape == (49, 2400)
    print("✅ Success!")