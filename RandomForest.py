"""
@author: KA LONG LEE (N9845097)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Spliting data to trainning and testing set
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.utils import shuffle

from dataset import generate_full_df

# train and return the random forest classifier
def Random_Forest_Prediction(X,y,estimators= 100):
    # Split the dataset to train and test
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)
    print("Random Forest Classifier is trainning.... *if estimator is too large it might take very long time...* \n")
    # Classifier setup and train
    rfc = RandomForestClassifier(n_estimators=estimators)
    rfc.fit(X_train,y_train)
    # Predict on test value
    yhat = rfc.predict(X_test)
    print("Accuracy on predicting train data : ")
    print(rfc.score(X_train,y_train))
    print("Accuracy on predicting test data : ")
    print(rfc.score(X_test,y_test))
    print("Classification Report : \n")
    print(classification_report(y_test, yhat) )
    return rfc

if __name__ == "__main__":
    print("Generating full information dataFrame ....")
    df = generate_full_df()
    print("Shuffling full information dataFrame ....")
    df = shuffle(df)
    X = df.drop(columns=['rating'], axis=1).values
    y = df.loc[:,'rating'].astype(str).values

    Random_Forest_Prediction(X,y,10)