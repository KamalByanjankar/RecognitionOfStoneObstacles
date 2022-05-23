import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def load_feature_data(filename):
    concat_data = []
    for file in filename:   
        print(file)
        required_features = pd.read_csv(file)
        concat_data.append(required_features)
    bigCobbleStone_data = pd.concat(concat_data, axis=0, ignore_index=True)
    return bigCobbleStone_data

def group_data_by_column_name(data, columnName):
    data = data.loc[data['Object Size'] == columnName].iloc[:, 6:]
    return data

def create_label(data, labelName):
    label = [labelName]*data.shape[0]
    return label

def split_data(data, label):
    train_X, test_X, train_y, test_y = train_test_split(data, label, test_size=0.30, random_state=42)
    return train_X, test_X, train_y, test_y

def normalization(train_test_data):
    normalized_data = preprocessing.normalize(train_test_data)
    return normalized_data

