from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
import glob
import operator

from PyQt6.QtWidgets import QFileDialog, QWidget, QApplication, QSizePolicy, QScrollArea, QMessageBox
from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem, QTableView, QVBoxLayout, QHeaderView
from glob import glob
from os.path import basename

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import random

from utils import feature_extraction, file_helper
from ui_files import MainWindow
from filter import DataProcessor
import pandas as pd
from utils.plot_canvas import PlotCanvas

import numpy as np
from sklearn.model_selection import train_test_split



class DataProcessingWindow(QMainWindow, MainWindow.Ui_MainWindow):
    def __init__(self, channel, *args, **kwargs):
        super(DataProcessingWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.channel = channel

        # Setup Operations
        self.btn_load_file.pressed.connect(self.load_file)
        self.btn_save_echo.pressed.connect(self.save_echo_to_file)
        self.files = None
        self.overall_echos = []

        # assign data_processing
        self.data_processing = DataProcessor()

        # parameter_frame
        # self.echo_size.textChanged.connect(self.update_echo_size)
        # self.noise_size.textChanged.connect(self.update_noise_size)
        self.plot_index.currentTextChanged.connect(
            self.add_plots)

        # load initialValues for Sample Frequency
        self.load_initialValues('1.953MHz')

        # canvas
        self.echo_plot = PlotCanvas(self, width=5, height=5, dpi=70, title="Echo data")
        self.echo_box.addWidget(self.echo_plot)

        self.time_domain_plot = PlotCanvas(
            self, width=5, height=5, dpi=70, title="Time Domain data")
        self.signal_box.addWidget(self.time_domain_plot)

        self.fft_plot = PlotCanvas(
            self, width=5, height=5, dpi=75, title="FFT data")
        self.fft_box.addWidget(self.fft_plot)

        # train Size
        self.txt_train_size.setText('70')
        # reg_ex = QRegularExpression("^[1-9][0-9]?$|^99$")
        # input_validator = QRegularExpressionValidator(reg_ex, self.txt_train_size)
        # self.txt_train_size.setValidator(input_validator)
        self.txt_train_size.textChanged.connect(
            self.on_train_text_value_changed)

        self.btn_data_open.setEnabled(False)
        self.btn_ml_open.pressed.connect(self.channel.open_ml_window)
        self.btn_ml_prediction_open.pressed.connect(
            self.channel.open_ml_prediction_window)
        self.btn_prediction_distance_open.pressed.connect(self.channel.open_prediction_distance_window)

        # self.show()
    
    def on_train_text_value_changed(self, value):
        if value and int(value) and int(value) < 100:
            self.txt_train_size.setText(value)

    def set_controller(self, controller):
        self.controller = controller

    def load_initialValues(self, freq):
        self.data_processing.change_selected_element(freq)
        self.echo_size.setText(str(self.data_processing.ECHO_SIZE))
        self.noise_size.setText(str(self.data_processing.NOISE_SIZE))

    def error_dailog(self, text):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Information)

        msg.setText(text)
        msg.setWindowTitle("Error")
        msg.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
        msg.exec()

    def update_echo_size(self, value):
        try:
            self.data_processing.ECHO_SIZE_LEFT = int(value)
        except:
            self.error_dailog('Value Should not have String')

    def update_noise_size(self, value):
        try:
            self.data_processing.ECHO_SIZE_LEFT = int(value)
        except:
            self.error_dailog('Value Should not have String')

    def load_file(self):
        try:
            directory = str(QFileDialog.getExistingDirectory(
                self, "Select Directory"))
            files = file_helper.files_from_directory(directory)
            
            self.load_files_in_table(
                sorted(files, key=lambda i: i['folder_name']))
        except:
            self.error_dailog('Please choose a file')

    def load_files_in_table(self, files):
        self.files = files
        self.table_widget.setRowCount(len(files))
        for row, item in enumerate(files):
            self.table_widget.setItem(
                row, 0, QTableWidgetItem(str(item['type'])))
            self.table_widget.setItem(
                row, 1, QTableWidgetItem(item['file_name']))
            self.table_widget.setItem(
                row, 2, QTableWidgetItem(item['model']))
            btn = QPushButton(f'Plot')
            btn.objectName = str(row)
            btn.clicked.connect(self.make_chart_with_file_index)
            self.table_widget.setCellWidget(row, 3, btn)

    def make_chart_with_file_index(self, file_name):
        try:
            import re
            # index = int(re.search(r'\d+', self.sender().text()).group())
            index = int(re.search(r'\d+', self.sender().objectName).group())

            self.time_domain_data_set = self.data_processing.get_time_domain_without_offset(
                self.files[index]['absolute_path'])

            self.plot_index.clear()
            for i in range(self.time_domain_data_set.shape[0]):
                self.plot_index.addItem(str(i))

        except:
            self.error_dailog('Chart cannot be created')

    def add_plots(self, value):
        try:
            if value:
                index = int(value)
                filtered_data_values = self.data_processing.get_filtered_values(
                    self.time_domain_data_set)
                echos_data = self.data_processing.get_echo_with_index(
                    filtered_data_values[index])
                self.time_domain_plot.plot(filtered_data_values[index])
                if len(echos_data) > 0:
                    self.echo_plot.plot(echos_data)
                    x, y = feature_extraction.fft_chart_value(
                        np.array(echos_data), self.data_processing.selected_element['value'], 
                        self.data_processing.config['low'], self.data_processing.config['high'])
                    self.fft_plot.plot_x_y(x, y, self.data_processing.config['low'], self.data_processing.config['high'])
                else:
                    self.echo_plot.plot([])
                    self.fft_plot.plot_x_y([], [])
        except:
            self.error_dailog('Combobox not ready, please Plot first')

    def get_features_from_echo(self, echos_data, row):
        df_fft = echos_data.iloc[:, 1:]
        fft_list = feature_extraction.fft_from_data_frame(
            df_fft, self.data_processing.selected_element['value'], 
            self.data_processing.config['low'], 
            self.data_processing.config['high'])
        fft_set = pd.DataFrame(fft_list)

        fft_set['model'] = row['model']
        fft_set['type'] = row['type']
        fft_set['distance'] = row['distance']
        fft_set['sample_frequency'] = self.sample_frequency.text()
        fft_set = fft_set.set_index(
            ['sample_frequency','distance', 'type', 'model']).reset_index()
        return fft_set

    def save_echo_to_file(self):
        echo_data_set = pd.DataFrame()
        fft_data_set = pd.DataFrame()
        for index, file in enumerate(self.files):
            if len(file['absolute_path'].split('/')) >= 4:
                time_domain_data_set = self.data_processing.get_time_domain_without_offset(
                    file['absolute_path'])
                filtered_data_values = self.data_processing.get_filtered_values(
                    time_domain_data_set)
                echos_data = self.data_processing.find_echos(
                    filtered_data_values)
                if isinstance(echos_data, pd.DataFrame):
                    row = {
                        'type': file['absolute_path'].split('/')[-4].upper(),
                        'model': file['absolute_path'].split('/')[-3],
                        'distance': file['file_name'].split('.')[0]
                    }
                    fft_data_set = fft_data_set.append(
                        self.get_features_from_echo(echos_data, row), ignore_index=True)

                    echos_data['model'] = row['model']
                    echos_data['type'] = row['type']
                    echos_data['sample_frequency'] = self.sample_frequency.text()
                    echos_data['distance'] = row['distance']
                    echos_data = echos_data.set_index(
                        ['sample_frequency', 'distance', 'type', 'model']).reset_index()
                    echo_data_set = echo_data_set.append(
                        echos_data, ignore_index=True)

        test_percentage = 1 - int(self.txt_train_size.text())/100
        if self.echo_file_name.text():
            X_train, X_test = train_test_split(echo_data_set, test_size=test_percentage, random_state=42)
            #X_train.reset_index().iloc[:,2:]
            config = self.data_processing.config
            freq = self.data_processing.selected_frequency
            file_name = self.echo_file_name.text()

            train_folder = f"{file_helper.executable}/{config['echo']}/{freq}/train/{file_name}.csv"
            test_folder = f"{file_helper.executable}/{config['echo']}/{freq}/test/{file_name}.csv"
            file_helper.check_or_create_folder(config, freq)

            X_train.sort_values(by=['distance']).reset_index().iloc[:,1:].to_csv(train_folder,index=False)
            X_test.sort_values(by=['distance']).reset_index().iloc[:,1:].to_csv(test_folder,index=False)

            train_folder = f"{file_helper.executable}/{config['feature']}/{freq}/train/{file_name}.csv"
            test_folder = f"{file_helper.executable}/{config['feature']}/{freq}/test/{file_name}.csv"

            X_train, X_test = train_test_split(fft_data_set, test_size=test_percentage, random_state=42)
            #X_train.reset_index().iloc[:,2:]
            X_train.sort_values(by=['distance']).reset_index().iloc[:,1:].to_csv(train_folder,index=False)
            X_test.sort_values(by=['distance']).reset_index().iloc[:,1:].to_csv(test_folder,index=False)
            return


    @pyqtSlot()
    def open_ml_window(self):
        self.machine_learning_window.show()
        self.close()

    @pyqtSlot()
    def open_ml_prediction_window(self):
        self.machine_learning_prediction_window.show()
        self.close()


if __name__ == '__main__':
    app = QApplication([])
    app.setApplicationName("Data Analysis")

    window = DataProcessingWindow()
    app.exec()
