import os
import cv2
import numpy as np
import tensorflow as tf 
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import load_model

path = os.path.abspath('test1')
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
X_train = X_train.reshape(X_train.shape[0], 64, 128, 1)
    


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

num_epochs = 20
batch_size = 40
# train model
input_shape = (64,128,1)
model_char_count = create_char_count_model(input_shape)
model_char_count.summary()
model_char_count.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_split=0.2)

# save the model
model_char_count.save('charcount_model.h5')
