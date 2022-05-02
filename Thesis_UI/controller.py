import sklearn
import sklearn.utils._cython_blas
import sklearn.tree._utils
#import sklearn.neighbors.quad_tree
#import sklearn.neighbors.typedefs
import pandas as pd
import pickle
import sklearn.tree
import sklearn.ensemble
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *

from PyQt6.QtWidgets import QFileDialog, QWidget, QApplication, QSizePolicy, QScrollArea, QMessageBox
from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem, QTableView, QVBoxLayout, QHeaderView
import numpy as np

from ui_files import Controller
from utils import file_helper
from machine_learning_training import MachineLearningWindow
from data_processing_window import DataProcessingWindow
from machine_learning_prediction import MachineLearningPredictionWindow
from prediction_distance import PredictionDistanceWindow

class ControllerWindow(QMainWindow, Controller.Ui_ControllerWindow):
    def __init__(self, *args, **kwargs):
        super(ControllerWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)

        self.btn_open.pressed.connect(self.open_window)

        self.data_window = DataProcessingWindow(self)
        self.machine_learning_window = MachineLearningWindow(self)
        self.machine_learning_prediction_window = MachineLearningPredictionWindow(
            self)
        self.prediction_distance_window = PredictionDistanceWindow(
            self)
        # show UI
        # if you want to show the controller UI
        # self.show() 

        # directly showing data_window
        # need to load config
        self.load_config()
        self.open_data_window()

    def error_dailog(self, text):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Information)

        msg.setText(text)
        msg.setWindowTitle("Error")
        msg.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
        msg.exec()

    def load_config(self):
        try:
            config = file_helper.get_config()
        except:
            self.error_dailog('Sorry, error in config file')
        try:
            file_helper.create_necessary_directories()
        except:
            self.error_dailog('Sorry, echos, features and model folder cannot be created')

    @pyqtSlot()
    def open_window(self):
        try:
            config = file_helper.get_config()
        except:
            self.error_dailog('Sorry, error in config file')
            return
        try:
            file_helper.create_necessary_directories()
            if self.radio_data_processing.isChecked():
                self.data_window.show()
            elif self.radio_ml_training.isChecked():
                self.machine_learning_window.show()
            elif self.radio_ml_predict.isChecked():
                self.machine_learning_prediction_window.show()
            else:
                self.error_dailog('Please select an option')
                return
            self.close()
        except:
            self.error_dailog('Sorry, echos, features and model folder cannot be created')

    @pyqtSlot()
    def open_ml_prediction_window(self):
        self.machine_learning_prediction_window.show()
        self.machine_learning_window.hide()
        self.prediction_distance_window.hide()
        self.data_window.hide()
        self.close()

    @pyqtSlot()
    def open_ml_window(self):
        self.machine_learning_window.show()
        self.machine_learning_prediction_window.hide()
        self.prediction_distance_window.hide()
        self.data_window.hide()
        self.close()

    @pyqtSlot()
    def open_data_window(self):
        self.data_window.show()
        self.machine_learning_window.hide()
        self.machine_learning_prediction_window.hide()
        self.prediction_distance_window.hide()
        self.close()

    @pyqtSlot()
    def open_prediction_distance_window(self):
        self.prediction_distance_window.show()
        self.machine_learning_window.hide()
        self.machine_learning_prediction_window.hide()
        self.data_window.hide()
        self.close()


if __name__ == '__main__':
    app = QApplication([])
    app.setApplicationName("Machine Learning")
    window = ControllerWindow()
    app.exec()
