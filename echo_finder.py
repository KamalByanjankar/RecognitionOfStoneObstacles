import numpy as np
import pandas as pd
from scipy import signal

NOISE_SIZE = 4000
ECHO_SIZE = 4096
ECHO_SIZE_LEFT = 1700
WINDOW_SIZE = 12384
THRESHOLD = 0.0003

def get_peak_value_using_rolling_window(data):
    data = np.array(data)
    number_of_windows = round(data.size / ECHO_SIZE)
    
    window_peak_locations = []
    window_peak_values = []

    for i in range(0, number_of_windows):
        window_peak = data[i * ECHO_SIZE : (i + 1) * ECHO_SIZE].max()
        if window_peak >= THRESHOLD:
            window_peak_location = i * ECHO_SIZE + data[i * ECHO_SIZE : (i+1)*ECHO_SIZE].argmax()
            if window_peak_location > ECHO_SIZE_LEFT:
                window_peak_locations.append(window_peak_location)
                window_peak_values.append(window_peak)
    
    if len(window_peak_locations):
        window_max_value = max(window_peak_values)  
        window_peak_point = data.tolist().index(window_max_value)
        
        peak_index = []
        for peak_value in range(window_peak_point, window_peak_point + 10):
            if data[peak_value].max() >= THRESHOLD:
                peak_index.append(peak_value)
            else:
                return None;
        return min(peak_index)
    else:
        return None
    
def get_echos(filtered_values):
    all_echo_range = []
    for index, data in enumerate(filtered_values):
        chopped_data = data[NOISE_SIZE:]
        max_point_distance = get_peak_value_using_rolling_window(chopped_data)
        if max_point_distance:
#             print(max_point_distance)
            cutting_distance = max_point_distance - ECHO_SIZE_LEFT
            if cutting_distance > 0 and max_point_distance + ECHO_SIZE_LEFT <= WINDOW_SIZE:
                echo_range = chopped_data[cutting_distance:]
                echo_range = echo_range[:ECHO_SIZE]
                all_echo_range.append(echo_range)
#     print(all_echo_range)
    return all_echo_range
