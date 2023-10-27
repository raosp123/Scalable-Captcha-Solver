import pandas as pd 
import os 
import tensorflow 
from tensorflow import keras
from keras.models import load_model
import numpy as np 
import cv2
import argparse

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--filespath', help='File with the images to use', type=str)
    parser.add_argument('--model-number-path', help='Path of the prediction model of number of char', type=str)
    parser.add_argument('--model1-path', help='Path of the first char model', type=str)
    parser.add_argument('--model2-path', help='Path of the second char model', type=str)
    parser.add_argument('--model3-path', help='Path of the third char model', type=str)
    parser.add_argument('--model4-path', help='Path of the fourth char model', type=str)
    parser.add_argument('--model5-path', help='Path of the fifth char model', type=str)
    parser.add_argument('--model6-path', help='Path of the sixth char model', type=str)
    args = parser.parse_args()
    
    if args.filespath is None:
        print("Please specify the files path")
        exit(1)
        
    if args.model_number_path is None:
        print("Please specify the path of the model that predicts the number of char")
        exit(1)
        
    if args.model1_path is None:
        print("Please specify the path of the first char model")
        exit(1)
        
    if args.model2_path is None:
        print("Please specify the path of the second char model")
        exit(1)
    
    if args.model3_path is None:
        print("Please specify the path of the third char model")
        exit(1)
    
    if args.model4_path is None:
        print("Please specify the path of the fourth char model")
        exit(1)
    
    if args.model5_path is None:
        print("Please specify the path of the fifth char model")
        exit(1)
        
    if args.model6_path is None:
        print("Please specify the path of the sixth char model")
        exit(1)

    def number_characters(model, image):
        image = cv2.imread(os.path.join(args.filespath,image))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image / 255.0
        
        prediction = model.predict(np.expand_dims(image, axis=0))
        number_char = round(prediction[0][0])
        
        if number_char > 6:
            number_char = 6
        
        return number_char

    def character_prediction(image, list_models, number_char):
        
        categories = ['#','%','+','-','0','1','2','3','4','5','6','7','8','9',"'",':','A','B','D','F','M','P','Q','R','T','U','V','W','X','Y','Z','[','\\',']','c','e','g','h','j','k','n','s','{','}']
        
        image = cv2.imread(os.path.join(args.filespath,image))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image / 255.0
        image = np.array(image)
        
        name = ''
        
        for i in range(number_char):
            index = np.argmax(list_models[i].predict(np.expand_dims(image, axis=0)))
            name = name + categories[index]
            
        return name

    list_images = os.listdir(args.filespath)

    model_nchar = load_model(args.model_number_path)
    char1_model = load_model(args.model1_path)
    char2_model = load_model(args.model2_path)
    char3_model = load_model(args.model3_path)
    char4_model = load_model(args.model4_path)
    char5_model = load_model(args.model5_path)
    char6_model = load_model(args.model6_path)

    list_models = [char1_model, char2_model, char3_model, char4_model, char5_model, char6_model]

    file_predictions = []

    for file in list_images:
        print(file)
        number_char = number_characters(model_nchar, file)
        file_predictions.append(character_prediction(file, list_models, number_char))

    df = pd.DataFrame()
    df['filename'] = list_images
    df['prediction'] = file_predictions

    df = df.sort_values(by= 'filename')
    df.to_csv('predictionsP2.csv', sep=',')   
     
    
if __name__ == '__main__':
    main()