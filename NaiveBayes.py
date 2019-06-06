"""
@author: KA LONG LEE (N9845097)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Spliting data to trainning and testing set
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

from dataset import generate_NB_df

def Naive_Bayes_Gaussian_Prediction(X,y,testSize = 0.2):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = testSize)

    print("Test data Size :")
    print(testSize)
    # Create Classifier
    nb = GaussianNB()

    print("Trainning Naive Bayes Classifier ...")
    # Train Classifier
    nb.fit(X_train, y_train)

    print("Accuracy on predicting train data : ")
    print(nb.score(X_train, y_train))
    print("Accuracy on predicting test data : ")
    print(nb.score(X_test, y_test))

    print("Classification Report : ")
    yhat = nb.predict(X_test)
    print(classification_report(y_test,yhat))

    return nb

if __name__ == "__main__":
    print("Generating Naive Bayes used DataFrame ....")
    df = generate_NB_df()
    X = df.drop(columns=['rating'], axis=1).values
    y = df['rating'].astype(str).values
    Naive_Bayes_Gaussian_Prediction(X,y)