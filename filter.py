from scipy.signal import butter, lfilter, freqz
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
    # y = lfilter(b, a, data)
    y = signal.filtfilt(b, a, data)

    b, a = highpass(30000, 1.953e6, order=5)
    y_ = signal.filtfilt(b, a, y)
    return y_