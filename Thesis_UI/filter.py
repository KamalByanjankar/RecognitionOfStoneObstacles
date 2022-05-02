import pandas as pd
import numpy as np

from scipy.signal import butter, lfilter, freqz
from scipy import signal

from utils.peaks_util import get_echo_peaks
from utils import file_helper
class DataProcessor:
    def __init__(self):
        self.config = file_helper.get_config()
        self.options = file_helper.get_config()['options']
        self.selected_frequency = list(self.options.keys())[0]
        self.selected_element = self.options[self.selected_frequency]
        self.LOW_PASS = self.config['low']
        self.HIGH_PASS = self.config['high']

        self.CHOPPED_WINDOW = self.config['choppedWindow']
        self.WINDOW_SIZE = self.config['windowSize']

        if 'low' in self.selected_element:
            self.LOW_PASS = self.selected_element['low']
        if 'high' in self.selected_element:
            self.HIGH_PASS = self.selected_element['high']
            
        self.NOISE_SIZE = self.selected_element['noise_size']
        self.DATA_HEADERS_SIZE = self.config['raw_data_header']
        self.ECHO_SIZE_LEFT = self.selected_element['echo_size_left']
        self.ECHO_SIZE = self.selected_element['echo_size']
        self.order = self.config['order']
        self.VARIANCE_THRESHOLD = self.selected_element['variance_threshold']

    def change_selected_element(self, element_name):
        if element_name in self.config['options']:
            self.selected_frequency = element_name
            self.selected_element = self.config['options'][element_name]
            self.ECHO_SIZE = self.selected_element['echo_size']
            self.NOISE_SIZE = self.selected_element['noise_size']
            self.ECHO_SIZE_LEFT = self.selected_element['echo_size_left']
            self.VARIANCE_THRESHOLD = self.selected_element['variance_threshold']
            if 'low' in self.selected_element:
                self.LOW_PASS = self.selected_element['low']
            if 'high' in self.selected_element:
                self.HIGH_PASS = self.selected_element['high']

    def butter_lowpass(self, cutoff, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_highpass(self, cutoff, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def apply_filter(self, data):
        b, a = self.butter_lowpass(
            self.LOW_PASS, self.selected_element['value'], order=self.order)
        # y = lfilter(b, a, data)
        y = signal.filtfilt(b, a, data)

        b, a = self.butter_highpass(
            self.HIGH_PASS, self.selected_element['value'], self.order)
        y_ = signal.filtfilt(b, a, y)
        return y_

    # Step 1 and 2
    # This method removes offset from time domain data
    # @params filename
    # @returns dataframe

    def get_time_domain_without_offset(self, filename):
        data_frame = pd.read_csv(filename, skiprows=[0], header=None)
        required_data_frame = data_frame.iloc[:, self.DATA_HEADERS_SIZE:]
        return required_data_frame.sub(required_data_frame.mean(axis=1), axis=0)

    # Step 3
    # this method applies low and high pass filter to time domain data
    # @params data_frame
    # @returns list

    def get_filtered_values(self, data_frame):
        new_data = []
        for data in data_frame.values:
            new_data.append(self.apply_filter(data))
        new_data = np.array(new_data)
        return new_data

    def get_echo_set_location(self, data):
        data = np.array(data)
        no_of_windows = round(data.size[1]/self.ECHO_SIZE)
        echo_set = []
        for d in data:
            window_peak_locations = []
            for i in range(0, no_of_windows):
                window_peak = d[i*self.ECHO_SIZE:(i+1)*self.ECHO_SIZE].max()
                if window_peak >= self.THRESHOLD:
                    window_peak_location = i*self.ECHO_SIZE + \
                        d[i*self.ECHO_SIZE:(i+1)*self.ECHO_SIZE].argmax()
                    if i > 0 or window_peak_location > self.ECHO_SIZE_LEFT:
                        window_peak_locations.append(window_peak_location)
            echos = []
            if len(window_peak_locations):
                echos = [window_peak_locations[0]]
                prev_echo = window_peak_locations[0]
                for w in window_peak_locations:
                    if prev_echo - w > self.ECHO_SIZE:
                        echos.append(w)
                    prev_echo = w
            echo_set.append(echos)
        return echo_set

    def get_echo_with_index(self, data_values):
        echo_list = []
        chopped_data = data_values[self.NOISE_SIZE:self.CHOPPED_WINDOW]
        max_point_distance = self.peak_value(chopped_data)
        #max_point_distance = get_peak_value_using_rolling_window(chopped_data)
        if max_point_distance:
            #print(max_point_distance)
            cutting_distance = max_point_distance - self.ECHO_SIZE_LEFT
            if cutting_distance > 0 and max_point_distance + self.ECHO_SIZE_LEFT <= self.WINDOW_SIZE:
                echo_list = chopped_data[cutting_distance:]
                echo_list = echo_list[:self.ECHO_SIZE]
        return echo_list

    def peak_value(self, data):
        data = np.array(data)    
        max_point_distance = 0
        peakData = 0

        max_point_distance = data.argmax()
        peakData = data.max()
        #print('Peak point: ', max_point_distance , ' and Peak value: ', peakData.max())
        if peakData > self.VARIANCE_THRESHOLD:
            return max_point_distance
        else:
            return None

    def find_echos(self, data_values):
        echo_range = []
        for value in data_values:
            individual_echo = self.get_echo_with_index(value)
            # if echo is not created it will not be inserted
            if len(individual_echo) > 0:
                echo_range.append(individual_echo)
        if len(echo_range) > 0:
            return pd.DataFrame(np.array(echo_range))
        return None
