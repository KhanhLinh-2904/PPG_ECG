import torch
import numpy as np
import pywt
from kymatio.torch import Scattering1D

class SignalPreprocessor:
    def __init__(self, signal_length, scattering_J=2, scattering_Q=8, dwt_wavelet='db4', dwt_level=3):
        """
        Args:
            signal_length (int): Length of the input signal (e.g., 2400 samples).
            scattering_J (int): Scale of scattering (controls the receptive field).
            scattering_Q (int): Number of filters per octave (frequency resolution).
            dwt_wavelet (str): Type of wavelet for DWT (e.g., 'db4', 'sym5').
            dwt_level (int): Decomposition level for DWT.
        """
        self.signal_length = signal_length
        self.dwt_wavelet = dwt_wavelet
        self.dwt_level = dwt_level
        
        # Initialize Kymatio Scattering Network
        # J=2 means the averaging scale is 2^2 = 4 samples. 
        # Since 2400 is divisible by 4, this works perfectly.
        self.scattering = Scattering1D(J=scattering_J, shape=(signal_length,), Q=scattering_Q)
        
        # Automatically calculate the number of channels scattering will produce
        # We run a dummy pass to get the output shape: (Batch, Channels, Time)
        dummy_input = torch.zeros(1, signal_length)
        self.scat_out_channels = self.scattering(dummy_input).shape[1]

    def process_ppg_scattering(self, ppg_batch):
        """
        Converts raw PPG signals to Wavelet Scattering features.
        
        Args:
            ppg_batch (torch.Tensor): Shape (Batch, Signal_Len)
            
        Returns:
            torch.Tensor: Shape (Batch, Scat_Channels, Reduced_Time_Len)
        """
        # Ensure input is normalized and correct type
        if not torch.is_tensor(ppg_batch):
            ppg_batch = torch.tensor(ppg_batch).float()
            
        # Kymatio scattering expects (Batch, Time)
        # Returns (Batch, Channels, Time_subsampled)
        ppg_batch_cpu = ppg_batch.detach().cpu()
        scattering_coeffs = self.scattering(ppg_batch_cpu)
        
        # To normalize features (often helpful for Neural Nets)
        # We take the log to make the distribution more Gaussian-like
        eps = 1e-6
        scattering_log = torch.log(torch.abs(scattering_coeffs) + eps)
        
        return scattering_log

    def process_ecg_dwt(self, ecg_batch):
        """
        Converts raw ECG signals to Discrete Wavelet Transform features.
        """
        batch_features = []
        
        # PyWavelets works on NumPy arrays
        ecg_np = ecg_batch.cpu().numpy() if torch.is_tensor(ecg_batch) else ecg_batch
        
        for signal in ecg_np:
            # Perform Discrete Wavelet Transform
            coeffs = pywt.wavedec(signal, self.dwt_wavelet, level=self.dwt_level)
            
            # Resize all coefficients to the size of the Approximation (lowest freq) band
            target_len = len(coeffs[0]) 
            resized_coeffs = []
            
            for c in coeffs:
                # Interpolate to match lengths
                c_tensor = torch.tensor(c).float().unsqueeze(0).unsqueeze(0) # (1, 1, Len)
                c_resized = torch.nn.functional.interpolate(c_tensor, size=target_len, mode='linear')
                resized_coeffs.append(c_resized.squeeze())
            
            # Stack features: Shape (Channels, Length)
            stacked = torch.stack(resized_coeffs) 
            batch_features.append(stacked)
            
        return torch.stack(batch_features) # Returns (Batch, Channels, Length)

# --- Usage Example ---
if __name__ == "__main__":

    # 1. Setup
    BATCH_SIZE = 4
    SIGNAL_LEN = 2400  # <--- UPDATED to 2400

    # Create the preprocessor
    # Note: Ensure that scattering_J is small enough such that 2**J <= SIGNAL_LEN
    # Here J=2 (default), so 2^2=4, which divides 2400 cleanly.
    preprocessor = SignalPreprocessor(signal_length=SIGNAL_LEN)

    # 2. Create Dummy Data (Raw Signals)
    raw_ppg = torch.randn(BATCH_SIZE, SIGNAL_LEN)
    raw_ecg = torch.randn(BATCH_SIZE, SIGNAL_LEN)

    # 3. Process the signals
    ppg_features = preprocessor.process_ppg_scattering(raw_ppg)
    ecg_features = preprocessor.process_ecg_dwt(raw_ecg)

    print("--- Preprocessing Results ---")
    print(f"Raw PPG Shape: {raw_ppg.shape}")
    print(f"Scattering Features (PPG) Shape: {ppg_features.shape}") 
    
    print(f"\nRaw ECG Shape: {raw_ecg.shape}")
    print(f"DWT Features (ECG) Shape: {ecg_features.shape}")