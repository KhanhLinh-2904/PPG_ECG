import numpy as np

def turningPointRatio(qrs_signal):
    qrs_signal = np.array(qrs_signal, dtype=float)
    qrs_length = len(qrs_signal)
    tp = np.zeros(qrs_length)

    # Find turning points
    for j in range(1, qrs_length - 1):
        if (qrs_signal[j - 1] < qrs_signal[j] > qrs_signal[j + 1]) or \
           (qrs_signal[j - 1] > qrs_signal[j] < qrs_signal[j + 1]):
            tp[j] = 1

    # Segment length
    l = len(tp)

    # Expected and actual values
    u_tp_expected = (2 * l - 4) / 3
    u_tp_actual = np.sum(tp)

    # Expected and real standard deviations
    sigma_tp_expected = np.sqrt((16 * l - 29) / 90)
    sigma_tp_real = np.std(tp)

    return u_tp_expected, u_tp_actual, sigma_tp_expected, sigma_tp_real
