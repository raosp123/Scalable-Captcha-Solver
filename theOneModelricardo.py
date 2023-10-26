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
#import tensorflow.keras as keras
from tensorflow import keras 
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

    categories = ['#','%','+','-','0','1','2','3','4','5','6','7','8','9',"'",':','A','B','D','F','M','P','Q','R','T','U','V','W','X','Y','Z','[','\\',']','c','e','g','h','j','k','n','s','{','}']
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

        for j, ch in enumerate(file_label):
            #print(y[i][j, :])
            train[j].append([gray_data, categories.index(ch)])
            #print(y[i][j])

    return train

#each input needs to be a grayscale for each pixel, so width * height * 1
def preProcessData(directory, width, height, captcha_symbols):

    #get list of files
    file_list = os.listdir(directory) #list of files in directory
    files = dict(zip(map(lambda x: x.split('.')[0], file_list), file_list)) #dict where the key is the label, and the value is the png

    #create X and Y for train/testing
    captcha_length = 6
    num_images = (len([name for name in os.listdir(directory)]))

    encoding_dict = {l:e for e,l in enumerate(captcha_symbols)}
    del encoding_dict['\n']
    decoding_dict = {e:l for l,e in encoding_dict.items()}

    print(encoding_dict)

    X = numpy.zeros((num_images, height, width)) #num of images, of height and width, with value each (grayscale)
    y = numpy.zeros((num_images, 43, 6))
    

    #temp = numpy.zeros((43, 5))#hold the one hot encoded 43*5 values before flattening

    for i, file_label in enumerate(files):
        
        file = files[file_label] 
        file_label = filename_format(reverse_dict, file_label)

        raw_data = cv2.imread(os.path.join(directory, file))
        gray_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2GRAY)
        gray_data = gray_data / 255.0

        #print(gray_data[0][0])

        # print(X[i][0])
        # print(gray_data[0])


        X[i] = gray_data

        for j, ch in enumerate(file_label):
            #print(y[i][j, :])
            y[i][encoding_dict[ch], j] = 1
            #print(y[i][j])

        #y[i] = temp.reshape(-1)


    #print(captcha_symbols.find('P'))
    #print(list(files)[0])

 
    #print(numpy.shape(y[0][0]))

    return X, y
    
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
    model.add(Dropout(0.2))
    model.add(Flatten())
    
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(0.001)))
    
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
    model.summary()
    
    return model


def create_model(captcha_length, captcha_num_symbols, input_shape, model_depth=5, module_size=2):
  input_tensor = keras.Input(input_shape)
  x = input_tensor
  for i, module_length in enumerate([module_size] * model_depth):
      print(f'module length is {module_length}, module size is {[module_size] * model_depth}, i = {i}')
      for j in range(module_length):
          x = keras.layers.Conv2D(32*2**min(i, 3), kernel_size=3, padding='same', kernel_initializer='he_uniform')(x)
          x = keras.layers.BatchNormalization()(x)
          x = keras.layers.Activation('relu')(x)
      x = keras.layers.MaxPooling2D(2)(x)

  x = keras.layers.Flatten()(x)
  x = [keras.layers.Dense(captcha_num_symbols, activation='softmax', name='char_%d'%(i+1))(x) for i in range(captcha_length)]
  model = keras.Model(inputs=input_tensor, outputs=x)

  model.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
                      metrics=['accuracy'])
  model.summary()
  return model

def splitY(trainY, num_images, captcha_len):

    trainYs = numpy.zeros((captcha_len, num_images, 43)) # there are 5 trainYs, the first one corresponds to 2000 images and the value for each character
    temp = numpy.zeros((num_images, captcha_len, 43))
    #trin y = 39999, 43, 5

    for i in range(num_images):
        
        temp[i] = trainY[i].transpose()


    # for i in range(captcha_len):
    #     for j in range(num_images):
    #         trainYs[i][j] = trainY[j].transpose()[i]

    return temp



if __name__ == '__main__':
   
    directory = 'test1'

    with open('symbols.txt') as symbols_file:
        captcha_symbols = symbols_file.readline()

    num_images = (len([name for name in os.listdir(directory)]))

    # file_list = os.listdir('trainvaluesfixed/')    
    test = ['#','%','+','-','0','1','2','3','4','5','6','7','8','9',"'",':','A','B','D','F','M','P','Q','R','T','U','V','W','X','Y','Z','[','\\',']','c','e','g','h','j','k','n','s','{','}']

    #print(test.index('\\'))
    # trainX, trainY = preProcessData(directory, 128, 64, captcha_symbols)

    # print(trainY.shape)

    # trainYs = splitY(trainY, num_images=len(trainX), captcha_len=5)



    # #trainY = trainY.reshape(num_images, 43 * 5)

    # print(trainYs.shape)


    train = trainingData(directory)
    random.shuffle(train)
    #print(train)

    #list of list implementation

    X=[]
    y=[]

    for i in range(6):
        X.append([])
        y.append([])

    for i in range(len(train)):
        for features, label in train[i]:
            X[i].append(features)
            y[i].append(label)


    print(numpy.array(X).shape)
  
    X = numpy.array(X)
    y = numpy.array(y)

    Y= numpy.zeros((6,40000,44))
    print(y.shape)
    #y_new = y.reshape(5,4000,44)

    for i in range(len(X)):
        X[i].reshape(-1, 64, 128, 1)
        Y[i] = to_categorical(y[i])

    # with numpy.printoptions(threshold=numpy.inf):
    #     print(Y[0])
    X=X.astype('float32')



    
    shape = (len(X[0][0]), len(X[0][0][0]), 1)

    print(shape)
    # # #print(f'The input image is {trainX[0]} and the output associated is {trainYs[0][0]}')

    first_model = createModel(X[0], shape, 44)
    print(Y[0].shape)
    print(X[0].shape)
    # second_model = create_model(5, 43, shape)
    batch_size = 64


    first_model.fit(X[0], Y[0], epochs=20, batch_size=batch_size, validation_split=0.2)

    first_model.save('please_work.h5')