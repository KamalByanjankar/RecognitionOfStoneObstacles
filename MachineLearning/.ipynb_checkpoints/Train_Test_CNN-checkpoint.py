from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Conv2D
from sklearn.model_selection import train_test_split
from ConfusionMatrix import create_confusion_matrix
from tensorflow.keras.utils import to_categorical
from MachineLearning_helper import split_data, normalization
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from ConfusionMatrix import create_confusion_matrix
import os


EPOCHS = 10
NUM_CLASSES = 3
BATCH_SIZE= 10


def train_CNN(data, label):
    label_encoded = label_Encoder(label)

    train_X, test_X, train_y, test_y = split_data(data, label_encoded)

    normalized_X_train = normalization(train_X)
    normalized_X_test = normalization(test_X)

    #converting list into matrix 
    train_X = np.array(normalized_X_train)[:,4:40].reshape(-1,6,6,1)
    test_X =  np.array(normalized_X_test)[:,4:40].reshape(-1,6,6,1)

    #converting number of objects as classes
    classes = np.unique(train_y)
    nClasses = len(classes)

    train_X = train_X.astype('float32')
    test_X = test_X.astype('float32')

    # Change the labels from categorical to one-hot encoding
    train_Y_one_hot = keras.utils.to_categorical(train_y)
    test_Y_one_hot = keras.utils.to_categorical(test_y)

    # Display the change for category label using one-hot encoding
#     print('Original label:', train_y[0])
#     print('After conversion to one-hot:', train_Y_one_hot[0])

#     print(train_Y_one_hot.shape)
#     print(test_Y_one_hot.shape)

    train_X,valid_X,train_label,valid_label = split_data(train_X, train_Y_one_hot)
#     np.shape(train_X),np.shape(valid_X), np.shape(train_label), np.shape(valid_label)
    
    model = create_CNN_model()

    #saving CNN model
#     checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath="./CNN_model.hdf5", verbose=1, save_best_only=True)
    #training CNN model
    model.fit(train_X, train_label, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(valid_X, valid_label))
#     , callbacks=[checkpointer])
    return model, test_X, test_Y_one_hot, test_y

def label_Encoder(label):
    label_encoder = LabelEncoder()
    label_encoded = label_encoder.fit_transform(label)
    return label_encoded

def create_CNN_model():
    #creating CNN model
    
    model = Sequential()
    
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', input_shape=(6,6,1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), padding='same'))
    
    model.add(Flatten())
    model.add(Dense(NUM_CLASSES, Activation('softmax')))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     model.summary()
    
    return model

def test_CNN_model(model, test_X, test_y):
    predicted_classes = model.predict(test_X)
    predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
    cm = confusion_matrix(predicted_classes, test_y)
    print(cm)
    print("Accuracy:", accuracy_score(predicted_classes, test_y))
    create_confusion_matrix(predicted_classes, test_y, 'CNN Confusion Matrix')

def evaluate_CNN_model(model, test_X, test_Y_one_hot):
#     model.load_weights('./CNN_model.hdf5')
    #evaluating model
    test_eval = model.evaluate(test_X, test_Y_one_hot)
    print(test_eval)
    
