import numpy as np

def shannonEntropy(signal, bin_size=16, window_size=128):
    signal = np.array(signal, dtype=float)

    counts, _ = np.histogram(signal, bins=bin_size)

    total_count = window_size - bin_size
    if total_count <= 0:
        return np.nan

    probabilities = counts / total_count

   
    entropy = 0
    for p in probabilities:
        if p > 0:
            entropy += p * (np.log(p) / np.log(1 / bin_size))  

    return entropy