from numpy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import numpy as np
import random

def fft_from_data_frame(data_frame):
    fs= 1.953e6
    random_idx = random.randrange(len(data_frame))
    print("Plot index", random_idx)
    row = data_frame.values[random_idx]
    print(row.shape)
    fft_data = fft(row)/row.size
    freq = fftfreq(row.size, d=1/fs)

    print(fft_data.shape)
    
    fft_data = np.abs(fft_data)
    plt.plot(freq, fft_data)
    plt.show()

    cut_high_signal = (fft_data).copy()
    cut_high_signal[(freq > 50000)] = 0
    cut_high_signal[(freq < 30000)] = 0
    signal_without_0 = list(filter(lambda a: a != 0, cut_high_signal))

    print(np.array(signal_without_0).shape)
    plt.axis([30000,50000, 0 , 0.015])
    plt.xlabel('Frequency')
    plt.ylabel('FFT')
    plt.plot(freq, cut_high_signal)
    plt.show()