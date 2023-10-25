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

'''
numpy printing cheat sheet.

array[#images][height][width][rgb/grayscale] =
        len(array) = number of images
        len(array[0]) = number of rows (since its a square, it it was non-uniform you would get number of columns in that row only)
        len(array[0][0]) = number of columns
        len(array[0][0][0]) = size of grayscale element

        array[0] is a 64*128 image with each cell being a single grayscale
        array[0][0] is image0, row0, therefore 128 columns of grayscale, which is the top row of the image 


output

    for each image we want the prediction for each character, so for character 0

    image * [
    
            1,0,0,0,0,0,0,0..... encoding of a character for all 44 characters
    ]
        
    
    image * number of symbols * encoding, so 


    y[0] = character 1 for the image which is something by 44


    number of images * captcha length * one hot encoding saying what character it is


    
Attempt 1:

    Train 5 models, each one is a classification problem for each individual letter in a captcha. So each image has 6 outputs, which a one hot encoding for each character, 
    where each model predicts the nth character in the captcha image. can our model predict the first character for each image?

        -> make a model for each character, so each model focuses on only one index in the captcha, we train the models with data that has at least the amount of characters
           that each model needs, e.g: a model that predicts the 3rd index of a captcha, needs to be trained on captchas of at least size 3, otherwise we are training it on blank space

    MAIN TODO: see parts 1 and 2 used as inspiration here (no code used from them): https://medium.com/@oneironaut.oml/solving-captchas-with-deeplearning-part-2-single-character-classification-ac0b2d102c96
        train a model to work to predict the first character, afterwards we move on to other indexes

    A. TODO 5 models for each character:
    
        1. read in file of image, for image put in y[image_number][i][j], where [j] is the value for the one hot encoding of the ith character:

            Visualisation

            X[image_number]=    0 . .   .   .   .   .   .   .   .   128           Y[image_number][captcha_index] for all captcha_index = 
                                .   [          GreyScale Values     ]                            captcha_index=0        [0,0,0,0,1.......0] = 44
                                .   [                               ]                             ....
                                64                                                               captcha_index=5        [0,1,0,0,0.......0] = 44

        2. make 5 models that classify on the nth character, so model1 predicts the 1st character

            -> model 1 would use the input value of the image, shape X[image_number]64*128*1(greyscale so 1), combined with y[image_number][model_number]
               so it would use the full X[image number] and the captcha index for that model, so model_number-1 (0, first index) if the model is predicting the 1st index

    B. TODO a model to predict how many characters are in a captcha, then feed them to the correct model in the previous part

        1. make a secondary y value in the preprocessing function, where it is just an encoding of what class the captcha belongs to, 

                Classes  1  2  3  4  5
                        [0, 0, 1, 0, 0] = captcha has 3 characters

                Then we can use this to make predictions on the testing data, if we get a captcha with 4 characters for example, we run it through models 1,2,3, and 4
                which predict the 1st, 2nd, 3rd, and 4th character respectively


'''



#each input needs to be a grayscale for each pixel, so width * height * 1
def preProcessData(directory, width, height, captcha_symbols):

    #get list of files
    file_list = os.listdir(directory)
    files = dict(zip(map(lambda x: x.split('.')[0], file_list), file_list))

    #create X and Y for train/testing
    captcha_length = 5




    num_images = (len([name for name in os.listdir('char2')]))


    X = numpy.zeros((num_images, height, width)) #num of images, of height and width, with value each (grayscale)
    y = numpy.zeros((num_images, captcha_length, 44))#we want to get for each image 5 rows with 1 columns so 2000*5*1, where the single column is one hot encoding of a single character


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
            y[i][j, captcha_symbols.find(ch)] = 1
            #print(y[i][j])


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
    
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=shape))
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

def splitY(trainY, num_images, captcha_len):

    trainYs = numpy.zeros((captcha_len, num_images, 44)) # there are 5 trainYs, the first one corresponds to 2000 images and the value for each character
    
    for i in range(captcha_len):
        for j in range(num_images):
            trainYs[i] = trainY[j][i]

    return trainYs


if __name__ == '__main__':
   

    with open('symbols.txt') as symbols_file:
        captcha_symbols = symbols_file.readline()

    file_list = os.listdir('char2')    

    trainX, trainY = preProcessData('char2', 128, 64, captcha_symbols)
    shape = (len(trainX[0]), len(trainX[0][0]), 1)

    trainYs = splitY(trainY, num_images=len(trainX), captcha_len=5) #training X is the same for every model, but trainYs[0] is the output for the image predicions of the first character
    
    print(trainYs[0].shape)

    #print(f'The input image is {trainX[0]} and the output associated is {trainYs[0][0]}')

    first_model = createModel(trainX, shape, 44)

    first_model.fit(trainX, trainYs[0], batch_size=40, epochs=2, validation_split=0.1)

    first_model.save('model_ch1.h5')