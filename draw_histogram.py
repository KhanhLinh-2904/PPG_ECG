import numpy as np
import matplotlib.pyplot as plt

# Load dataset
data = np.load("datasets/MIMIC_test.npz", allow_pickle=True)
ecgs = data["ppgs"]  # shape: (num_samples, length)
labels = data["labels"]

print("ecg: ", ecgs.shape)
# Flatten all ECG signals into one array
all_signals = ecgs.reshape(-1)
print("ecg reshape: ", all_signals.shape)

# Compute mean and variance
mean_val = np.mean(all_signals)
var_val = np.var(all_signals)

print("Overall Mean:", mean_val)
print("Overall Variance:", var_val)

# Plot histogram of all ECG values
plt.figure(figsize=(8, 5))
plt.hist(all_signals, bins=100, color="blue", alpha=0.7, edgecolor="black")
plt.title("Histogram of All PPG Signals")
plt.xlabel("Amplitude")
plt.ylabel("Frequency")
plt.grid(True, alpha=0.3)

# Add mean line
plt.axvline(mean_val, color="red", linestyle="dashed", linewidth=2, label=f"Mean = {mean_val:.4f}")
plt.legend()
plt.show()
