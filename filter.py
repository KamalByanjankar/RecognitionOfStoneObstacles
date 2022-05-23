from scipy.signal import butter
from scipy import signal

def lowpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def highpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def apply_filter(data):
    b, a = lowpass(50000, 1.953e6, order=5)
    y = signal.filtfilt(b, a, data)

    b, a = highpass(30000, 1.953e6, order=5)
    y_ = signal.filtfilt(b, a, y)
    return y_

def apply_filter_data_set(data_set):
    filtered_data = []
    for d in data_set:
        filtered_data.append(apply_filter(d))
    return filtered_data