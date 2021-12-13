import pandas as pd
from sklearn import preprocessing

def load_features_data(filename):
    features_data = pd.read_csv(filename)
    return features_data

def normalize_train_X(train_X):
    normalized_X_train = preprocessing.normalize(train_X)
    return normalized_X_train

def normalize_test_X(test_X):
    normalized_X_test = preprocessing.normalize(test_X)
    return normalized_X_test