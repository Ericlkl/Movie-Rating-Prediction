#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 09:27:49 2019

@author: jenny
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# Spliting data to trainning and testing set
from sklearn.model_selection import train_test_split, GridSearchCV
# Fitting Multiple Linear Regression to the trainning set
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from sklearn import tree

# Matplotlib setting 

def DT_evaluation(clf,x_axis,n_folds,X_train,X_test,y_train,y_test):
    
    scores = clf.cv_results_['mean_test_score']
    scores_std = clf.cv_results_['std_test_score']
    std_error = scores_std / np.sqrt(n_folds)
    
    plt.figure()
    plt.plot(x_axis, scores + std_error, 'b--o', markersize=3)
    plt.plot(x_axis, scores - std_error, 'b--o', markersize=3)
    plt.plot(x_axis, scores,color='black', marker='o',  
             markerfacecolor='blue', markersize=5)
    plt.fill_between(x_axis, scores + std_error, scores - std_error, alpha=0.2)
    plt.xlabel('Maximum tree depth')
    plt.ylabel('Cross validation score +/- std error')
    plt.title('Cross validation results')

    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)
    
    print('Classification report for training data: \n', classification_report(y_train, pred_train))
    print('Classification report for test data: \n',  classification_report(y_test, pred_test))
    print('The best choice of depth: ' + str(clf.best_params_['max_depth']))
    # source: https://scikit-learn.org/stable/auto_examples/exercises/plot_cv_diabetes.html#sphx-glr-auto-examples-exercises-plot-cv-diabetes-py


if __name__ == "__main__":

    # DATA PREPROCESSING
    # Import dataset
    rating_dataset = pd.read_csv('./data/ratings.csv')
    movies_dataset = pd.read_csv("./data/movies.csv")
    
    # Set the index by movieId, This line of code only able to execute once
    movies_dataset.set_index('movieId', inplace = True)
    
    # Convert genres to dummy variable dataset
    # movies_genres_dummy <- a dataframe contains the genres values for all the movies and using movieId as index
    movies_genres_dummy = movies_dataset['genres'].str.get_dummies(sep='|')
    # Remove (no genres listed) from dummy because all 0 can represent (no genres listed)
    movies_genres_dummy = movies_genres_dummy.drop(columns=["(no genres listed)"], axis=1)
    
    
    # Use this If you think you are able to work on the years 
    # Cannot use year as parameter because in movie 3xxxx there is a movie does not have years
    # movies_dataset["year"] = movies_dataset["title"].str.extract(r"\(([0-9]+)\)").astype(dtype=np.int)
    
    # Pre Processing for KMeans Algorithm finding the similar movies
    movies_dataset = movies_dataset.drop(columns=["genres"], axis=1)
    movies_dataset = pd.merge(movies_dataset, movies_genres_dummy, on = 'movieId', how = "left")
    
    full_rating_dataset = pd.merge(rating_dataset[["userId","movieId","rating"]], movies_genres_dummy, on = 'movieId', how = "left")
    
    full_rating_dataset['rating'] = full_rating_dataset['rating'] * 2
    full_rating_dataset['rating'] = full_rating_dataset['rating'].astype(int)
    
    genres = full_rating_dataset.iloc[:,3:].columns
    
    
    
    # DATA EXTRACTION
    X_df = full_rating_dataset.drop(columns=['rating'], axis=1)
    y_df = full_rating_dataset.iloc[:,2]
    X = X_df.values
    y = y_df.values
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)
    max_depth = 20
    
    clf = tree.DecisionTreeClassifier()
    tuned_parameters = [{'max_depth': np.arange(1,max_depth+1)}]
    
    clf = GridSearchCV(clf,tuned_parameters,cv=5)
    clf = clf.fit(X_train, y_train)
    
    '''
    score:
    Returns the mean accuracy on the given test data and labels.

    In multi-label classification, this is the subset accuracy which is a harsh
    metric since you require for each sample that each label set be correctly predicted.
    '''
    train_score = clf.score(X_train,y_train)
    print(train_score)
    test_score = clf.score(X_test,y_test)
    print(test_score)
    
    yhat_test = clf.predict(X_test)
    yhat_train = clf.predict(X_train)
    train_error = mean_squared_error(yhat_train,y_train)
    print(train_error)
    test_error = mean_squared_error(yhat_test,y_test)
    print(test_error)
    
    DT_evaluation(clf,np.arange(1,max_depth+1),5,X_train,X_test,y_train,y_test)
    
    










