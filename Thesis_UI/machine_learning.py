import pandas as pd
import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from utils import file_helper
class MachineLearningHelper:
    def __init__(self):
        self.config = file_helper.get_config()
        self.options = file_helper.get_config()['options']
        self.selected_frequency = list(self.options.keys())[0]
        self.selected_element = self.options[self.selected_frequency]

        self.clf_ready = False
        self.clf_loaded = None
        self.select_classifier()

    def change_selected_element(self, element_name):
        if element_name in self.options:
            self.selected_frequency = element_name
            self.selected_element = self.options[element_name]
            self.select_classifier()

    def select_classifier(self):
        a_type = self.selected_element['MLP']
        solver = a_type['solver']
        activation = a_type['activation']
        hidden_layer_sizes = tuple(a_type['hidden_layer_sizes'])
        self.clf_mlp = MLPClassifier(solver=solver, activation=activation, max_iter=500,
                                     alpha=1e-5, hidden_layer_sizes=hidden_layer_sizes, random_state=12)

    def train(self, X_train, y_train):
        self.clf_mlp.fit(X_train, y_train)
        self.clf_ready = True

    def predict(self, X_test):
        result_mlp = self.clf_mlp.predict(X_test)
        return result_mlp, []

    def save_model(self):
        from datetime import datetime
        date_str = str(datetime.now()).replace(':','').replace('.','').replace(' ','')
        config =  file_helper.get_config()
        file_mlp = f"{file_helper.executable}/{config['model']}/mlp/{date_str}_{self.selected_frequency}"
        pickle.dump(self.clf_mlp, open(file_mlp, 'wb'))

    def load_model(self, model):
        self.clf_loaded = model
        self.clf_ready = True

    def loaded_predict(self, X_test):
        result = self.clf_loaded.predict(X_test)
        return result
