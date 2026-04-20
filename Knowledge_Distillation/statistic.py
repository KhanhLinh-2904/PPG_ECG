import numpy as np
import matplotlib.pyplot as plt

# train_data_path = "processed_data/mimic3_v1_train.npz"

test_data_path = "processed_data/MIT_BIH_train_segments.npz"
# data_train = np.load(train_data_path, allow_pickle=True)
data_test = np.load(test_data_path, allow_pickle=True)
# ppgs = data_test["ppgs"]
records = data_test["records"]
unique_records = np.unique(records)
print("unique_names: ", unique_records)




labels = data_test["labels"]
index_af = 0
index_norm = 0
# print("len ppgs: ", len(ppgs))
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