import numpy as np
import matplotlib.pyplot as plt

#ECG Signal
train_data_path = "datasets/total_min_max_train.npz"
test_data_path = "datasets/total_min_max_test.npz"
data_train = np.load(train_data_path, allow_pickle=True)
data_test = np.load(test_data_path, allow_pickle=True)
labels = data_test["labels"]

index_af = 0
index_norm = 0

for i in range(len(labels)):
    if i == 0:
        print("labels: ", labels[i])
    if labels[i] == 1:
        index_af += 1
    else:
        index_norm += 1

print("index_af: ", index_af)
print("index_norm: ", index_norm)
print("Total: ", len(labels))