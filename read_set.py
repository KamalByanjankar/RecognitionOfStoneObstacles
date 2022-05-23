import numpy as np
import pandas as pd
from echo_finder import get_echos
from fft_implementation import fft_from_data_frame, get_fft_list
from filter import apply_filter
from file_helper import load_data_points, get_time_domain_data_without_offset, check_NaN_in_dataFrame
from scipy import signal

def get_echo_from_object(file_list, distance_list):
    data_with_offset = get_raw_data(file_list, distance_list)
    data_without_offset = get_without_offset(data_with_offset)
    filtered_data = get_filtered_data(data_without_offset)
    echo_set = get_echo_set(filtered_data)
    return echo_set

# def get_filtered_data_from_file(file_list, distance_list):
#     data_with_offset = get_raw_data(file_list, distance_list)
#     data_without_offset = get_without_offset(data_with_offset)
#     filtered_data = get_filtered_data(data_without_offset)
#     return filtered_data

def get_raw_data(folder_set, distance_set):
    sub_folder_set = [1]

    data_with_offset = []
    for i, folder_name in enumerate(folder_set):
        folder = folder_set[i]

        for i, sub_folder_name in enumerate(sub_folder_set):
            sub_folder = sub_folder_set[i]

            for distance in distance_set:
#                 print(distance)
                filename = './rawData/dataSet/{}/{}/{}.csv'.format(folder, sub_folder, distance)
                time_domain_data_frame = load_data_points(filename)
                required_time_domain_data_frame = check_NaN_in_dataFrame(time_domain_data_frame)
    #             plot_graphs(np.array(time_domain_data_frame), 2)
                data_with_offset.append(required_time_domain_data_frame)
                print(filename, required_time_domain_data_frame.shape)
    return data_with_offset

def get_without_offset(data_with_offset):
    data_without_offset = []
    for i, data in enumerate(data_with_offset):
        required_data_without_offset = get_time_domain_data_without_offset(data)
        data_without_offset.append(required_data_without_offset)
    return data_without_offset
        
def get_filtered_data(data_without_offset):
    filtered_data = []
    for i, data in enumerate(data_without_offset):
        data_after_filter = apply_filter(data)
        filtered_data.append(data_after_filter)
    return filtered_data

def get_echo_set(filtered_data):
    echo_set = []
    for i, data in enumerate(filtered_data):
        required_echos = get_echos(data)
        echo_set = echo_set + required_echos
    return echo_set

def print_aggregate(data):
    print('MIN:', np.min(np.amax(data, axis=1)))
    print('MAX:', np.max(np.amax(data, axis=1)))
    print('AVG:', np.average(np.amax(data, axis=1)))
    
def print_height(ref_data, HEIGHT, data):
    ref_data = np.array(ref_data)[:, 8:34]
#     print(ref_data.shape)
#     MIN = np.min(np.amax(ref_data, axis=1))
    MAX = np.max(np.amax(ref_data, axis=1))
#     AVG = np.average(np.amax(ref_data, axis=1))
    newlist = [x for x in data if x != 'nan']
    newData = np.array(newlist)[:, 8:34]
    print(np.shape(newData))
    print('Max Amplitude:', "{:.8f}".format(np.max(np.amax(newData, axis=1))))
#     height_min = HEIGHT * np.min(np.amax(newData, axis=1)) / MIN
    height_max = HEIGHT * np.max(np.amax(newData, axis=1)) / MAX * 11.5
#     height_avg = HEIGHT * np.average(np.amax(newData, axis=1)) / AVG 
#     print('HEIGHT MIN=', "{:.2f}".format(height_min))
    print('HEIGHT MAX=', "{:.2f}".format(height_max))
#     print('HEIGHT AVG=', "{:.2f}".format(height_avg))
    

