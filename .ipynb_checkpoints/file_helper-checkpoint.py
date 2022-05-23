import pandas as pd

def load_data_points(filename):
    data_frame = pd.read_csv(filename, skiprows=[0], header=None)
    required_data_frame = data_frame.iloc[:, 1:]
    return required_data_frame

def get_time_domain_data_without_offset(data_frame):
    time_domain_data_without_offset = data_frame.sub(data_frame.mean(axis=1), axis=0).values
    return time_domain_data_without_offset

def check_NaN_in_dataFrame(data_frame):
    nan_df = data_frame[data_frame.isna().any(axis=1)]
    if len(nan_df) > 0:
        data_frame = data_frame.dropna(how='all')
        return data_frame
    else:
        return data_frame