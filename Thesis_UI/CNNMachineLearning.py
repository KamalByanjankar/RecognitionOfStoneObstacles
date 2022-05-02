import pandas as pd
import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Conv2D
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from utils import normalize

from utils import file_helper
class CNNMachineLearning:
    def __init__(self):
        self.config = file_helper.get_config()
        self.clf_ready = False
        self.clf_loaded = None

    def setup(self, X, y, train_size):
        test_percentage = 1 - train_size/100
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y_encoded, test_size=test_percentage, random_state=42)
        classes = np.unique(self.y_train)
        # model is ready
        self.cnn_setup(len(classes))

    def cnn_setup(self, classes):
        a_type = self.config['options']['1.953MHz']['CNN']
        self.cnnModel = Sequential()
        self.cnnModel.add(Conv2D(a_type["hidden_layer"], kernel_size=(3, 3), padding='same', input_shape=(6,6,1)))
        self.cnnModel.add(Activation('relu'))
        self.cnnModel.add(MaxPooling2D(pool_size=(3, 3), padding='same'))
        self.cnnModel.add(Flatten())
        self.cnnModel.add(Dense(classes, Activation('softmax')))
        self.cnnModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.cnnModel.summary()
        self.clf_ready = True


    def train(self):
        a_type = self.config['options']['1.953MHz']['CNN']
        # change target to category
        y_train_category = to_categorical(self.y_train)

        # convert X_train to matrix form
        X_normalize = normalize.custom_normalization(self.X_train)
        X_train_cnn = self.cnn_X_conversion(X_normalize)
        X_train_cnn = X_train_cnn.astype('float32')

        # finally get train and validate data
        X_train,X_validate,y_train,y_validate = train_test_split(X_train_cnn, y_train_category, test_size=0.30, random_state=13)
        # fitting the model
        self.cnnModel.fit(X_train, y_train, epochs=a_type['epochs'], batch_size=a_type['batch_size'], verbose=1, validation_data=(X_validate, y_validate))
        
    def predict(self):
        # change target to category
        y_train_category = to_categorical(self.y_test)

        # convert X_train to matrix form
        X_test_normalize = normalize.custom_normalization(self.X_test)
        X_test_cnn = self.cnn_X_conversion(X_test_normalize)
        X_test_cnn = X_test_cnn.astype('float32')

        test_eval = self.cnnModel.evaluate(X_test_cnn, y_train_category, verbose=0)
        predicted_classes = self.cnnModel.predict(X_test_cnn)
        predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
        return predicted_classes

    def cnn_X_conversion(self, X_data):
        return np.array(X_data)[:,4:40].reshape(-1,6,6,1)

    def save_model(self):
        # save_model
        from datetime import datetime
        date_str = str(datetime.now()).replace(':','').replace('.','').replace(' ','')
        file_path= f"{file_helper.executable}/{self.config['model']}/CNN/{date_str}"
        # checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=file_path + "/weights.hdf5", verbose=1, save_best_only=True)
        self.cnnModel.save(file_path+'cnnModel.hd5')
        return

    def load_model(self, model):
        self.cnnModel = tf.keras.models.load_model(model)

        self.clf_loaded = model
        self.clf_ready = True
        return

    def loaded_predict(self, X_test):
        X_test_cnn = self.cnn_X_conversion(X_test)
        predicted_classes = self.cnnModel.predict(X_test_cnn)
        predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
        print(predicted_classes)
        return predicted_classes

    def get_categorical(self, y):
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        #  target to category
        return to_categorical(y_encoded)