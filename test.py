import numpy as np
import matplotlib.pyplot as plt

#ECG Signal
test_data_path = "/home/linhhima/Pre_processing_data/datasets/MIMIC_SegmentSplit_70_15_15_test.npz"
data = np.load(test_data_path, allow_pickle=True)
ecgs = data["ecgs"]
labels = data["labels"]
index_af = 0
index_norm = 0
print('len: ', ecgs.shape)

for i in range(len(ecgs)):
    if i == 0:
        print("labels: ", labels[i])
    if labels[i] == 1:
        index_af += 1
    else:
        index_norm += 1

print("index_af: ", index_af)
print("index_norm: ", index_norm)
print("Total: ", len(labels))