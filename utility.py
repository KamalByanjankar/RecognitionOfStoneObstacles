import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from matplotlib.pyplot import figure
import numpy as np

THRESHOLD = 0.0027
SPEED_OF_SOUND = 34400 # cm/s

NOISE_SIZE = 4000
#
# data is a single row for which peak is to be calculated
# when plot is true it will print the figure with peaks marked as x above threshold
#
def get_distance(data, threshold=THRESHOLD, plot=False):
    row_ = np.array(data)
    x = row_
    peaks, _ = find_peaks(x, height=threshold)
    # it has a list of peaks above threshold
    peaks = peaks[peaks>NOISE_SIZE]
    first_peak = peaks[0]
    if plot:
        figure(figsize=(20, 18), dpi=80)
        plt.plot(x)
        plt.plot(peaks, x[peaks], "x")
        plt.plot(np.zeros_like(x), "--", color="gray")
        plt.show()
        
    time_of_flight = first_peak/(1.953e6)
    # distance in cm
    distance = SPEED_OF_SOUND * time_of_flight
    return distance