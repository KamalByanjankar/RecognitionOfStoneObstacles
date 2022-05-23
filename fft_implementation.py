from numpy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd

def fft_from_data_frame(data_frame):
    fs= 1.953e6
    row = data_frame.values[0]
    fft_data = fft(row)/row.size
    freq = fftfreq(row.size, d=1/fs)

    print(row.shape)
    
    fft_data = np.abs(fft_data)
    plt.plot(freq, fft_data)
    plt.show()
    
    print(max(fft_data))
    
    cut_high_signal = (fft_data).copy()
    cut_high_signal[(freq > 50000)] = 0
    cut_high_signal[(freq < 30000)] = 0
    signal_without_0 = list(filter(lambda a: a != 0, cut_high_signal))
    
    print(np.array(signal_without_0).shape)
    plt.axis([30000,50000,0,0.02])
    plt.xlabel('Frequency')
    plt.ylabel('FFT')
    plt.plot(freq, cut_high_signal)
    plt.show()
    #     df = DataFrame(cut_high_signal)
    
def get_fft_list(data_frame):
    fs= 1.953e6
    
    signal_set = []
    for row in data_frame.values:
        fft_data = fft(row)/row.size
        freq = fftfreq(row.size, d=1/fs)
        cut_high_signal = np.abs(fft_data).copy()
        cut_high_signal[(freq > 50000)] = 0
        cut_high_signal[(freq < 30000)] = 0
        signal_without_0 = list(filter(lambda a: a != 0, cut_high_signal))
        signal_set.append(np.abs(signal_without_0))
    return signal_set

def get_fft_from_data_set(data_set): 
    required_echo_data = pd.DataFrame(data_set)
    fft_list = get_fft_list(required_echo_data)
    return fft_list