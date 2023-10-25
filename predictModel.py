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
import json
from sklearn.metrics import confusion_matrix, classification_report
#import tensorflow.keras as keras
from tensorflow import keras 
from myModel import filename_format, preProcessData, splitY

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
We want to get an X with all the test images, and a y with all the correct values, then compare the values
'''

def decodeOutputs(preds, symbols):

    #loop through prediction and image file list to see if chars match
    isFound = False
    #for i in 



if __name__ == '__main__':
   
    with open('symbols.txt') as symbols_file:
        captcha_symbols = symbols_file.readline()

    file_list = os.listdir('testvaluesfixed/')    

    testX, testY = preProcessData('testvaluesfixed/', 128, 64, captcha_symbols)

    #testYs = splitY(testY, num_images=len(testX), captcha_len=5)

    model = keras.models.load_model('model_ch1')

    preds = model.predict(testX)
  
    print(preds)
    print(preds.shape)
    numpy.savetxt('output.txt', preds)











