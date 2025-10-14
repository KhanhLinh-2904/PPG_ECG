import numpy as np

# Load dataset
test_data_path = "datasets/ECG_MIT-BIH_data.npz"
data = np.load(test_data_path, allow_pickle=True)

labels = data["labels"]
ecgs = data["ecgs"]

# Find indices of each class
non_af_indices = np.where(labels == 0)[0]
af_indices = np.where(labels == 1)[0]

# Randomly select 5635 from each class
selected_non_af = np.random.choice(non_af_indices, size=8538)
selected_af = np.random.choice(af_indices, size=8538)

# Combine indices
selected_indices = np.concatenate([selected_non_af, selected_af])

# Shuffle indices to mix 0s and 1s
np.random.shuffle(selected_indices)

# Extract the signals and labels
selected_ecgs = ecgs[selected_indices]
selected_labels = labels[selected_indices]

print("Shape of selected ECGs:", selected_ecgs.shape)
print("Label distribution:", np.unique(selected_labels, return_counts=True))

# Save into a new npz file
save_path = "datasets/ECG_MIT-BIH_data_new.npz"
np.savez(save_path, ecgs=selected_ecgs, labels=selected_labels)

print(f"Saved balanced dataset (5635 per class) to {save_path}")
