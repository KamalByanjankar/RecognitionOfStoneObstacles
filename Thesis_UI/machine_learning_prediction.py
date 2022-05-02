from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
import glob
import operator
from sklearn.model_selection import train_test_split
from PyQt6.QtCore import QRegularExpression
from PyQt6.QtGui import QRegularExpressionValidator
from PyQt6.QtWidgets import QFileDialog, QWidget, QApplication, QSizePolicy, QScrollArea, QMessageBox
from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem, QTableView, QVBoxLayout, QHeaderView
from glob import glob
from os.path import basename
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import random

from utils import feature_extraction, normalize, file_helper
from machine_learning import MachineLearningHelper
import pandas as pd
from utils.plot_canvas import PlotCanvas
from ui_files import MachineLearningPrediction
from CNNMachineLearning import CNNMachineLearning

import numpy as np
import pickle
from collections import Counter


class MachineLearningPredictionWindow(QMainWindow, MachineLearningPrediction.Ui_MachineLearningPrediction):

    def __init__(self, channel, *args, **kwargs):
        super(MachineLearningPredictionWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)

        self.config = file_helper.get_config()

        # Setup Operations
        self.btn_load_file.pressed.connect(self.load_file)
        self.channel = channel
        self.files = None
        self.data_set = {}
        # assign data_processing
        self.machine_learning_processor = MachineLearningHelper()
        self.cnn_machine_learning = CNNMachineLearning()

        # radio
        self.radioCNN.setChecked(True)
        self.model = 'CNN'
        self.radioMLP.pressed.connect(self.setMLP)
        self.radioCNN.pressed.connect(self.setCNN)

        # setup combobox
        self.cb_output.addItems(list(self.config['output'].keys()))

        # canvas
        self.prediction_plot_canvas = PlotCanvas(
            self, width=2, height=2, dpi=50, title="Predicted Result")
        self.plot_prediction.addWidget(self.prediction_plot_canvas)

        self.prediction_confusion_matrix_canvas = PlotCanvas(
            self, width=2, height=2, dpi=70, title="Predicted Confusion Matrix")
        self.plot_confusion_matrix.addWidget(
            self.prediction_confusion_matrix_canvas)

        # self.btn_load_file.pressed.connect(self.start_process)
        self.btn_start_prediction.pressed.connect(
            self.on_start_prediction_press)
        self.btn_load_model.pressed.connect(self.select_model)

        # Feature text 
        self.txt_feature_size.setDisabled(True)

        # navigation
        self.btn_ml_prediction_open.setEnabled(False)
        self.btn_ml_open.pressed.connect(self.channel.open_ml_window)
        self.btn_prediction_distance_open.pressed.connect(self.channel.open_prediction_distance_window)
        self.btn_data_open.pressed.connect(
            self.channel.open_data_window)


    def setCNN(self):
        self.radioCNN.setChecked(True)
        self.radioMLP.setChecked(False)
    
    def setMLP(self):
        self.radioCNN.setChecked(False)
        self.radioMLP.setChecked(True)

    def arrange_data(self):
        data = self.data_set['all_data']['data']
        self.data_set['all_data']['X'] = data.iloc[:, self.config['feature_header']:].values.tolist()
        self.data_set['all_data']['y'] = data['type'].values.tolist()

    def on_start_prediction_press(self):
        try:
            if self.data_set:
                self.arrange_data()
                X_normalized_predict = normalize.custom_normalization(self.data_set['all_data']['X'])
                if self.model == 'CNN':
                    result = self.cnn_machine_learning.loaded_predict(X_normalized_predict)
                else:
                    result = self.machine_learning_processor.loaded_predict(X_normalized_predict)
                    labels = [*Counter(result).keys()]  # equals to list(set(words))
                    values = [*Counter(result).values()]
                    self.add_result_to_table(labels, values)
                self.predicted_result = result
                from sklearn.preprocessing import LabelEncoder
                label_encoder = LabelEncoder()

                if self.model == 'CNN':
                    labels = self.config['output'][self.cb_output.currentText()]
                    label_encoder.fit_transform(labels)
                    label_encoded_result = label_encoder.inverse_transform(self.predicted_result)
                    self.prediction_confusion_matrix_canvas.plot_confusion_matrix(label_encoded_result, self.data_set['all_data']['y'], labels)
                    labels = [*Counter(label_encoded_result).keys()]  # equals to list(set(words))
                    values = [*Counter(label_encoded_result).values()]
                    self.prediction_plot_canvas.plot_prediction(labels, values)                        
                    self.add_result_to_table(labels, values)
                else:
                    self.prediction_plot_canvas.plot_prediction(labels, values)
                    self.prediction_confusion_matrix_canvas.plot_confusion_matrix(
                        self.predicted_result, self.data_set['all_data']['y'], self.machine_learning_processor.clf_loaded.classes_.tolist())
            else:
                self.error_dailog('Please select training files')
        except Exception as e :
            self.error_dailog('Error in processing')

    def add_result_to_table(self, labels, values):
        # adding in the table
        counter = 1
        self.table_widget.setRowCount(len(labels)+1)
        for index, item in enumerate(labels):
            self.table_widget.setItem(
                counter, 0, QTableWidgetItem(item))
            self.table_widget.setItem(
                counter, 1, QTableWidgetItem(str(values[index])))
            counter = counter + 1

    def error_dailog(self, text):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Information)

        msg.setText(text)
        msg.setWindowTitle("Error")
        msg.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
        msg.exec()

    def success_dialog(self, text):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Information)

        msg.setText(text)
        msg.setWindowTitle("Success")
        msg.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
        msg.exec()

    def select_model(self):
        try:
            if (self.radioMLP.isChecked()):
                filename = QFileDialog.getOpenFileName(self, 'Open File')
                model = pickle.load(open(filename[0], 'rb'))
                self.machine_learning_processor.load_model(model)
                self.model = 'MLP'
                self.txt_model_name.setText(filename[0])
            else:
                filename = QFileDialog.getExistingDirectory(self, 'Open File')
                self.cnn_machine_learning.load_model(filename)
                self.model = 'CNN'
                self.txt_model_name.setText(filename)
        except:
            self.error_dailog(
                'Invalid model file')

    def load_and_plot_result(self):
        try:
            files = QFileDialog.getOpenFileNames(
                self, 'Open File', 'csv')
            result = []
            for f in files[0]:
                data = pd.read_csv(f)
                if 'type' in data.columns:
                    result = result + data.type.values.tolist()
            if self.model == 'CNN':
                self.prediction_confusion_matrix_canvas.plot_confusion_matrix(
                    self.predicted_result, result, list(range(self.cnn_machine_learning.cnnModel.output_shape[1])))
            else:
                self.prediction_confusion_matrix_canvas.plot_confusion_matrix(
                    self.predicted_result, result, self.machine_learning_processor.clf_loaded.classes_.tolist())
        except:
            self.error_dailog(
                'Invalid model file')

    def load_file(self):
        try:
            config = file_helper.get_config()
            files = QFileDialog.getOpenFileNames(
                self, 'Open File', 'csv')
            self.data_set = {}

            r = None
            for f in files[0]:
                data = pd.read_csv(f)
                if 'type' in data.columns:
                    d = data
                else:
                    if data.shape[1] > 5000:
                        self.error_dailog('Please do prediction first')
                        return
                    else:
                        d = data.iloc[:, :]
                if 'all_data' not in self.data_set:
                    self.data_set['all_data'] = {}
                    self.data_set['all_data']['data'] = d
                else:
                    self.data_set['all_data']['data'] = pd.concat([self.data_set['all_data']['data'], d])
            self.load_files_in_table()
        except Exception as e:
            print(e)
            self.error_dailog('Please choose a file')

    def load_files_in_table(self):
        files = []
        self.table_widget.setRowCount(0)
        for item in self.data_set:
            files.append({
                'name': item,
                'data_size': self.data_set[item]['data'].shape[0],
            })

        shape = self.data_set[item]['data'].iloc[:, self.config['feature_header']:].shape[1]
        self.txt_feature_size.setText(str(shape))

        self.table_widget.setRowCount(len(files))
        for row, item in enumerate(files):
            self.table_widget.setItem(
                row, 0, QTableWidgetItem(item['name']))
            self.table_widget.setItem(
                row, 1, QTableWidgetItem(str(item['data_size'])))

    @pyqtSlot()
    def open_ml_window(self):
        self.machine_learning_window.show()
        self.close()

    @pyqtSlot()
    def open_data_window(self):
        self.data_window.show()
        self.close()


if __name__ == '__main__':
    app = QApplication([])
    app.setApplicationName("Machine Learning Prediction")

    window = MachineLearningPredictionWindow()
    window.show()
    app.exec()
