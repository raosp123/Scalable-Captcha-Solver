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
#import tensorflow.keras as keras
from tensorflow import keras 

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

    MAIN TODO: see parts 1 and 2 here: https://medium.com/@oneironaut.oml/solving-captchas-with-deeplearning-part-2-single-character-classification-ac0b2d102c96
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
    symbols_txt='symbols.txt'
    symbols_length = 44


    num_images = (len([name for name in os.listdir('trainvaluesfixed')]))

    X = numpy.zeros((num_images, height, width, 1)) #num of images, of height and width, with value each (grayscale)
    y = numpy.zeros((num_images, captcha_length, 44))#we want to get for each image 5 rows with 1 columns so 2000*5*1, where the single column is one hot encoding of a single character

    # for j, ch in enumerate:

    print(numpy.shape(y))
    print((y[0][0]))

    # for i in length:

    #     X[]


    #print(f'X= {X}, shape = {numpy.shape(X)}, type = {type(X)}')
    
class ImageSequence(keras.utils.Sequence):
    def __init__(self, directory_name, batch_size, captcha_length, captcha_symbols, captcha_width, captcha_height):
        self.directory_name = directory_name
        self.batch_size = batch_size
        self.captcha_length = captcha_length
        self.captcha_symbols = captcha_symbols
        self.captcha_width = captcha_width
        self.captcha_height = captcha_height

        file_list = os.listdir(self.directory_name)
        self.files = dict(zip(map(lambda x: x.split('.')[0], file_list), file_list))
        self.used_files = []
        self.count = len(file_list)

    def __len__(self):
        return int(numpy.floor(self.count / self.batch_size))

    def __getitem__(self, idx):
        X = numpy.zeros((self.batch_size, self.captcha_height, self.captcha_width, 3), dtype=numpy.float32)
        y = [numpy.zeros((self.batch_size, len(self.captcha_symbols)), dtype=numpy.uint8) for i in range(self.captcha_length)]

        #print(f'remaining files is {len(self.files.keys())}')
        for i in range(self.batch_size):
            if len(self.files.keys()) > 0:
                random_image_label = random.choice(list(self.files.keys()))
                random_image_file = self.files[random_image_label]

                # We've used this image now, so we can't repeat it in this iteration
                self.used_files.append(self.files.pop(random_image_label))

                # We have to scale the input pixel values to the range [0, 1] for
                # Keras so we divide by 255 since the image is 8-bit RGB
                raw_data = cv2.imread(os.path.join(self.directory_name, random_image_file))
                rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
                processed_data = numpy.array(rgb_data) / 255.0
                X[i] = processed_data

                random_image_label = filename_format(reverse_dict, random_image_label)

               

                for j, ch in enumerate(random_image_label):

                    #print(f'j = {j}, character = {ch}') #debugging

                    y[j][i, :] = 0
                    y[j][i, self.captcha_symbols.find(ch)] = 1

                    #print(y)

            else:
                exit

        return X, y
    

if __name__ == '__main__':
   

    with open('symbols.txt') as symbols_file:
        captcha_symbols = symbols_file.readline()

    preProcessData('trainvaluesfixed/', 128, 64, captcha_symbols)