import os
import cv2
import numpy as np
import tensorflow as tf 
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import load_model
import argparse


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--width', help='Width of captcha image', type=int)
    parser.add_argument('--height', help='Height of captcha image', type=int)
    parser.add_argument('--files-path', help='File with the images to use', type=str)
    parser.add_argument('--epochs', help='Number of epochs', type=int)
    parser.add_argument('--batch-size', help='Number of batch size', type=int)
    args = parser.parse_args()
    
    if args.width is None:
        print("Please specify the captcha image width")
        exit(1)

    if args.height is None:
        print("Please specify the captcha image height")
        exit(1)
        
    if args.files_path is None:
        print("Please specify the captcha image files path")
        exit(1)
        
    if args.epochs is None:
        print("Please specify the number of epochs")
        exit(1)
        
    if args.batch_size is None:
        print("Please specify the batch size")
        exit(1)
    
    path = os.path.abspath(args.files_path)
    images = os.listdir(path)
    
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

    def filename_format(dictionary, filename):

        #print(f'The image label is: {filename}\n')

        for key in dictionary:
            filename = filename.replace(key, dictionary[key])

        #print(f'The new image label is: {filename}\n')

        return filename

    X_train = []
    y_train = []
    for filename in images:
        image_path = os.path.join(path, filename)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        X_train.append(img)
        if filename.endswith('.png'):
            captcha_text = os.path.splitext(filename)[0]
            captcha_text = filename_format(reverse_dict, captcha_text)
            y_train.append(len(captcha_text))
        
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_train = X_train / 255.0
    X_train = X_train.reshape(X_train.shape[0], args.height, args.width, 1)
        


    def create_char_count_model(input_shape):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(1, activation='linear'))  # Output a single number for character count prediction

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])  # Use mean squared error for regression
        return model


    # train model
    input_shape = (args.height,args.width,1)
    model_char_count = create_char_count_model(input_shape)
    model_char_count.summary()
    model_char_count.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, validation_split=0.2, callbacks=[keras.callbacks.ModelCheckpoint("bestncharcomplex.h5", save_best_only=True)])

    # save the model
    model_char_count.save('charcount_model.h5')
    
    
if __name__ == '__main__':
    main()

