"""
Original file is located at
    https://colab.research.google.com/drive/1Dd7BhozTjiquLs_QKwzIwrJW8hRvwrRG
"""


import numpy as np
import cv2
import numpy as np
from sklearn import metrics
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from functions import *


def main():
    #Trains a system to recognise the text of a train image and then predict the text of the input image and returns and also
    #evaluate the accuracy of the system to the whole image 
    #the size of descriptors for each level of contours
    
    #N=[50,25,25]
    #N=[80,40,40]
    N=[300,150,150]
    #N=[100,50,50]
    # choose the image on which the system will be trained on and load it
    train_text_file = "text1_v3.txt"
    train_image_file='text1_v3.png'
    train_image = cv2.imread(train_image_file,0)
    #create the datasets on which the classifiers will be trained
    dataset = create_datasets(train_image,train_text_file,N,'utf-8')
    #extract the features from each dataset and flatten them
    X1, y1 = extract_features(dataset[1])
    X2, y2 = extract_features(dataset[2])
    X3, y3 = extract_features(dataset[3])
    X1=flattenTrainSet(X1)
    X2=flattenTrainSet(X2)
    X3=flattenTrainSet(X3)
    #split the datasets in train and test set and train one classifier on the train set of each one
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=42)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=42)
    #if the dataset 3 is smaller than the number of neighboors,create some extra data and train the classifier
    #for the case of text1.png where there are no many letters with 3 contours
    if X3.shape[0]<4:
      x3_ext=np.zeros([5,X3.shape[1]])
      X3_train=np.full_like(x3_ext,X3[0].reshape(1,-1))
      X3_test=np.full_like(x3_ext,X3[0].reshape(1,-1))
      y3_ext=np.empty([5,1],dtype='U')
      y3_train=np.full_like(y3_ext,y3[0])
      y3_test=np.full_like(y3_ext,y3[0])
      classifier3 = train_classifier(X3_train, y3_train)
    else:
      X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.3, random_state=42)
      classifier3 = train_classifier(X3_train, y3_train)
    classifier1 = train_classifier(X1_train, y1_train)
    classifier2 = train_classifier(X2_train, y2_train)
    #evaluate the classifier on the test sets
    print("RESULT FOR TEST SET OF TRAIN IMAGE :",train_image_file)
    confusion_matrix1, weighted_accuracy1 = evaluate_classifier(classifier1, X1_test, y1_test)
    confusion_matrix2, weighted_accuracy2 = evaluate_classifier(classifier2, X2_test, y2_test)
    confusion_matrix3, weighted_accuracy3 = evaluate_classifier(classifier3, X3_test, y3_test)

    #choose the image you want to predict its text 
    #predict the text in the input image
    test_text_file = "text2.txt"
    test_image_file="text2_150dpi_rot.png"
    image = cv2.imread(test_image_file, 0)
    predicted_text=readtext(image,classifier1,classifier2,classifier3,N)
    #print the predicted text
    for line_text in predicted_text:
      print(line_text)
    #create the dataset on which we will test the perfomance
    dataset = create_datasets(image,test_text_file,N,'utf-16')
    #extract the features from each dataset and flatten them
    X1, y1 = extract_features(dataset[1])
    X2, y2 = extract_features(dataset[2])
    X3, y3 = extract_features(dataset[3])
    X1=flattenTrainSet(X1)
    X2=flattenTrainSet(X2)
    X3=flattenTrainSet(X3)
    #evaluate the classifier on the whole dataset
    print("RESULT FOR all the TEST IMAGE :",test_image_file)
    confusion_matrix1, weighted_accuracy1 = evaluate_classifier(classifier1, X1, y1)
    confusion_matrix2, weighted_accuracy2 = evaluate_classifier(classifier2, X2, y2)
    confusion_matrix3, weighted_accuracy3 = evaluate_classifier(classifier3, X3, y3)


if __name__ == "__main__":
    main()
