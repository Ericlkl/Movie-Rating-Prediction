"""
Created on Wed May 22 13:58:35 2019

@author: jenny
        KA LONG LEE (N9845097)
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from dataset import generate_user_avg_rating_df

def KMeans_SSE_Graphic(X,Max_K):
    # Performs K = 1 to 50 for clustering the data
    Ks = range(1,Max_K)
    # Array for saving inertias values for each K Clustering
    sse = []

    # Loop K times 
    for k in Ks:
        # Create KMeans
        model = KMeans(n_clusters = k)
        # Train KMeans 
        model.fit(X)
        sse.append(model.inertia_)

    # Plot inertias values vs K numbers 
    plt.ylabel("SSE")
    plt.xlabel("Numbers of K")
    plt.title("K-Means Algorithm on Clustering Movies")
    plt.plot(Ks,sse,'-o')

def KMeans_Cluster(df,X,K,fileName):
    # Create KMeans Algorithm to cluster the data in 20 groups 
    model = KMeans(n_clusters = K)

    # Train KMeans
    model.fit(X)

    # Get the cluster values for each movie
    labels = model.predict(X)

    # Put the cluster value back to the dataset
    df['cluster'] = labels

    # Save it into a new csv file, you can view the result in this csv file
    df.sort_values(by=['cluster']).to_csv(fileName)

    print("Success!")
    return labels