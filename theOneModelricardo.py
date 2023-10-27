#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import cv2
import numpy
import string
import random
import argparse
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from keras import layers, regularizers
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU


import pandas as pd
from keras.utils import to_categorical

reverse_dict = {
    'hash': '#',
    'backslash': '\\',
    'exclamation': '!',
    'forwardslash': '/',
    'lbrace': '{',
    'rbrace': '}',
    'pipe': '|',
    'doublequote': '"',
    'singlequote': "'",
    'question': '?',
    'at': '@',
    'backtick': '`',
    'colon': ':'
}

def trainingData(directory):

    categories = ['#','%','+','-','0','1','2','3','4','5','6','7','8','9',"'",':','A','B','D','F','M','P','Q','R','T','U','V','W','X','Y','Z','[','\\',']','c','e','g','h','j','k','n','s','{','}','&']
    file_list = os.listdir(directory) #list of files in directory
    files = dict(zip(map(lambda x: x.split('.')[0], file_list), file_list))

    train = []
    for i in range(6):
        train.append([])

    for i, file_label in enumerate(files):
        
        file = files[file_label] #file.png
        file_label = filename_format(reverse_dict, file_label)

        raw_data = cv2.imread(os.path.join(directory, file))
        gray_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2GRAY)
        gray_data = gray_data / 255.0

        #print(gray_data[0][0])

        # print(X[i][0])
        # print(gray_data[0])
        h = 0
        while h < 6:
            if h < (len(file_label)-1):
                for j, ch in enumerate(file_label):
                    #print(y[i][j, :])
                    train[j].append([gray_data, categories.index(ch)])
                    h = h +1
                    #print(y[i][j])
            else:
                train[h].append([gray_data, 44])
                h = h+1

    return train
    
# converts the filename back to character form
def filename_format(dictionary, filename):

    #print(f'The image label is: {filename}\n')

    for key in dictionary:
        filename = filename.replace(key, dictionary[key])

    #print(f'The new image label is: {filename}\n')

    return filename

def createModel(trainX, shape, num_classes):

    model = keras.Sequential()
    
    model.add(Conv2D(32, (2, 2), activation='relu', input_shape=shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (2, 2), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (2, 2), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(0.001)))
    
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
    model.summary()
    
    return model

def createModelComplex(trainX, shape, num_classes):

    model = keras.Sequential()
    
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(0.001)))
    
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
    model.summary()
    
    return model


def smallerCaptchaArray(index):
    X = []
    y = []

    for features, label in train[index]:
            X.append(features)
            y.append(label)


    X = numpy.array(X)
    y = numpy.array(y)

    Y= numpy.zeros((40000,44))

    print(y.shape)
    #y_new = y.reshape(5,4000,44)

    
    X.reshape(-1, 64, 128, 1)
    Y = to_categorical(y)

    print(Y.shape)
    print(X.shape)

    X.reshape(-1, 64, 128, 1)
    Y = to_categorical(y)

    X=X.astype('float32')

    shape = (len(X[0]), len(X[0][0]), 1)

    return X, Y, shape

def main():

#python theOneModelricardo.py --data_dir=trainvalues1ch/ --batch_size=64 --epochs=25 --model_index=1 --validation_split=0.2
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='select input directory for training set', type=int)
    parser.add_argument('--batch_size', help='select batch size', type=int)
    parser.add_argument('--epochs', help='select number of epochs', type=int)
    parser.add_argument('--model_index', help='select what character index you want the model to predict', type=int)
    parser.add_argument('--validation_split', help='select what percent of the input data will be set aside for validation', type=int)


    args = parser.parse_args()

    if args.data_dir is None:
        print("select input directory for training set")
        exit(1)

    if args.batch_size is None:
        print("select batch size")
        exit(1)

    if args.epochs is None:
        print("select number of epochs")
        exit(1)

    if args.model_index is None:
        print("select what character index you want the model to predict")
        exit(1)

    if args.validation_split is None:
        print("select what percent of the input data will be set aside for validation")
        exit(1)

    directory = f'trainvalues{args.model_index}ch'
    train = trainingData(args.data_dir)
    
    for i in range(len(train)): 
        random.shuffle(train[i])

    #list of list implementation

    X,Y,shape = smallerCaptchaArray((args.model_index-1))

    first_model = createModelComplex(X, shape, 44)
    batch_size = 64

    print(f'traing model {args.model_index}')

    first_model.fit(X, Y, epochs=25, batch_size=args.batch_size, validation_split=0.2, callbacks=[keras.callbacks.ModelCheckpoint("ch1-e{epoch}.h5", save_best_only=True)])

    first_model.save(f'character{args.model_index}.h5')

if __name__ == '__main__':
   
    #main()


    #Debugging
    model_num = 4

    directory = f'trainvalues{model_num}ch'
    train = trainingData(directory)
    
    for i in range(len(train)): 
        random.shuffle(train[i])

    #list of list implementation

    X,Y,shape = smallerCaptchaArray((model_num-1))

    first_model = createModelComplex(X, shape, 44)
    batch_size = 64

    print(f'traing model {model_num}')

    first_model.fit(X, Y, epochs=35, batch_size=batch_size, validation_split=0.2, callbacks=[keras.callbacks.ModelCheckpoint("ch1-e{epoch}.h5")])

    first_model.save(f'character{model_num}.h5')