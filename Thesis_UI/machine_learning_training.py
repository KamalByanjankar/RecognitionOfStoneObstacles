from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
import glob
import operator
from sklearn.model_selection import train_test_split
from PyQt6.QtCore import QRegularExpression, pyqtSignal
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

from CNNMachineLearning import CNNMachineLearning
import pandas as pd
from ui_files import MachineLearning
from utils.plot_canvas import PlotCanvas
import numpy as np

class MachineLearningWindow(QMainWindow, MachineLearning.Ui_MachineLearning):
    def __init__(self, channel, *args, **kwargs):
        super(MachineLearningWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.channel = channel
        # Setup Operations
        self.btn_load_file.pressed.connect(self.load_file)
        self.files = None
        self.data_set = {}
        # assign data_processing
        self.machine_learning_processor = MachineLearningHelper()
        self.cnn_machine_learning = CNNMachineLearning()

        # load initial values for sample frequency
        self.load_initialValues('1.953MHz')

        # train Size
        self.txt_train_size.setText('70')
        reg_ex = QRegularExpression("^[1-9][0-9]?$|^99$")
        input_validator = QRegularExpressionValidator(reg_ex, self.txt_train_size)
        self.txt_train_size.setValidator(input_validator)
        self.txt_train_size.textChanged.connect(
            self.on_train_text_value_changed)

        # canvas
        self.mlp_plot_canvas = PlotCanvas(
            self, width=10, height=10, dpi=60, title="MLP Confusion Matrix")
        self.mlp_plot.addWidget(self.mlp_plot_canvas)

        self.rf_plot_canvas = PlotCanvas(
            self, width=10, height=10, dpi=60,  title="CNN Confusion Matrix")
        self.rf_plot.addWidget(self.rf_plot_canvas)

        # self.btn_load_file.pressed.connect(self.start_process)
        self.btn_train.pressed.connect(self.on_train_press)
        self.btn_save_model.pressed.connect(self.save_model)

        # Feature text 
        self.txt_feature_size.setDisabled(True)

        self.btn_ml_open.setEnabled(False)
        self.btn_ml_prediction_open.pressed.connect(
            self.channel.open_ml_prediction_window)
        self.btn_data_open.pressed.connect(
            self.channel.open_data_window)
        self.btn_prediction_distance_open.pressed.connect(self.channel.open_prediction_distance_window)

    def set_controller(self, controller):
        self.controller = controller

    def load_initialValues(self, freq):
        self.machine_learning_processor.change_selected_element(freq)

    def on_train_text_value_changed(self, value):            
        if value and int(value) and int(value) < 100:
            self.txt_train_size.setText(value)
            try:
                self.split_data()
                self.load_files_in_table()
            except:
                self.error_dailog('Please load the file.')

    def on_train_press(self):
        if self.data_set:
            X_train = []
            X_test = []
            y_train = []
            y_test = []
            for item in self.data_set:
                X_train = X_train + self.data_set[item]['X_train']
                X_test = X_test + self.data_set[item]['X_test']
                y_train = y_train + self.data_set[item]['y_train']
                y_test = y_test + self.data_set[item]['y_test']
            labels = [*self.data_set]
            X_normalized_train = normalize.custom_normalization(X_train)
            X_normalized_test = normalize.custom_normalization(X_test)

            self.cnn_machine_learning.setup(X_train+X_test, y_train+y_test, int(self.txt_train_size.text()))
            self.cnn_machine_learning.train()
            result_cnn = self.cnn_machine_learning.predict()

            self.machine_learning_processor.train(X_normalized_train, y_train)
            result_mlp, result_rf = self.machine_learning_processor.predict(
                X_normalized_test)
            self.mlp_plot_canvas.plot_confusion_matrix(
                result_mlp, y_test, labels)
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            label_encoder = label_encoder.fit(labels)
            result_cnn_later = label_encoder.inverse_transform(result_cnn)
            result_cnn_y = label_encoder.inverse_transform(self.cnn_machine_learning.y_test)

            self.rf_plot_canvas.plot_confusion_matrix(
                result_cnn_later, result_cnn_y, labels)
            return
        else:
            self.error_dailog('Please select training files')

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

    def save_model(self):
        if self.machine_learning_processor.clf_ready:
            try:
                self.machine_learning_processor.save_model()
                self.cnn_machine_learning.save_model()
                self.success_dialog(
                    'Model is successfully saved')
            except:
                self.error_dailog(
                    'Model destination folder does not exist')
        else:
            self.error_dailog(
                'Model is not ready, please train the model first')

    def load_file(self):
        try:
            config = file_helper.get_config()
            files = QFileDialog.getOpenFileNames(
                self, 'Open File', 'csv')

            self.data_set = {}
            dataframe = None
            for f in files[0]:
                data = pd.read_csv(f)
                dataframe = pd.concat([dataframe, data])

            types = dataframe.type.unique()
            types.sort()
            for t in types:
                d = dataframe.loc[dataframe['type'] == t].iloc[:, config['feature_header']:]
                if t not in self.data_set:
                    self.data_set[t] = {}
                    self.data_set[t]['data'] = d.values.tolist()
                else:
                    self.data_set[t]['data'] = self.data_set[t]['data'] + \
                        d.values.tolist()
            self.split_data()
            self.load_files_in_table()
        except:
            self.error_dailog('Please choose a file')

    def load_files_in_table(self):
        files = []
        for item in self.data_set:
            files.append({
                'name': item,
                'data_size': len(self.data_set[item]['data']),
                'train_size': len(self.data_set[item]['X_train']),
                'test_size': len(self.data_set[item]['X_test'])
            })
        
        shape = len(self.data_set[item]['X_train'][0])
        self.txt_feature_size.setText(str(shape))
        self.table_widget.setRowCount(len(files))
        for row, item in enumerate(files):
            self.table_widget.setItem(
                row, 0, QTableWidgetItem(item['name']))
            self.table_widget.setItem(
                row, 1, QTableWidgetItem(str(item['data_size'])))
            self.table_widget.setItem(
                row, 2, QTableWidgetItem(str(item['train_size'])))
            self.table_widget.setItem(
                row, 3, QTableWidgetItem(str(item['test_size'])))

    def split_data(self):
        test_percentage = 1 - int(self.txt_train_size.text())/100

        for item in self.data_set:
            self.data_set[item]['X_train'], self.data_set[item]['X_test'], self.data_set[item]['y_train'], self.data_set[item]['y_test'] = train_test_split(
                self.data_set[item]['data'], [item] * len(self.data_set[item]['data']), test_size=test_percentage, random_state=42)

    def custom_normalization(self, X_set):
        new_X_set = []
        for X in X_set:
            min = np.min(X)
            max = np.max(X)
            value = max - min
            data_set = []
            for data in X:
                data_set.append(((data - min) / value) + 0)
            new_X_set.append(data_set)
        return new_X_set


if __name__ == '__main__':
    app = QApplication([])
    app.setApplicationName("Data Analysis")

    window = MachineLearningWindow()
    app.exec()
