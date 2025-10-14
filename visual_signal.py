import numpy as np
import matplotlib.pyplot as plt

#ECG Signal
test_data_path = "datasets/MIMIC_test.npz"
data = np.load(test_data_path, allow_pickle=True)
ecgs = data["ecgs"]
labels = data["labels"]
print('len: ', len(ecgs))
for i in range(len(ecgs)):
    if labels[i] == 1:
        index_af = i
    else:
        index_norm = i

ecg_af = ecgs[index_af]
ecg_norm = ecgs[index_norm]
# print("ecg: ", ecgs[0])
N = len(ecg_af)      # số mẫu trong ecg
fs = 125    # Hz (tần số lấy mẫu, bạn cần thay đúng giá trị dataset của mình)
t = np.arange(N)/fs

# Create figure
plt.figure(figsize=(12, 6))

# --- Plot Normal ECG ---
plt.subplot(2, 1, 1)
plt.plot(t, ecg_norm, color='tab:blue')
plt.title("Normal ECG Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")
plt.grid(True)

# --- Plot AF ECG ---
plt.subplot(2, 1, 2)
plt.plot(t, ecg_af, color='tab:red')
plt.title("Atrial Fibrillation (AF) ECG Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")
plt.grid(True)

plt.tight_layout()
plt.savefig("ECG_AF_vs_Normal_Subplots.png", dpi=300)
plt.show()