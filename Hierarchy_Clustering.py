"""
@author: KA LONG LEE (N9845097)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.cluster.hierarchy import linkage, fcluster,dendrogram
from dataset import generate_user_avg_rating_df

def draw_user_dendrogram():

    print("Generating user dataframe for Hierarchy Clustering ...")
    user_avg_rating_df = generate_user_avg_rating_df()
    X = user_avg_rating_df.iloc[:,1:].values

    mergings = linkage(X, method='complete')

    print("Drawing User Dendrogram ...")
    dendrogram(
        mergings,
        labels = range(0,611),
    )

    plt.show()

    return mergings

def HC_cluster_user(mergings, height):
    print("Clustering user group and generating labels...")
    labels = fcluster(mergings, height, criterion = 'distance')
    
    user_avg_rating_df = generate_user_avg_rating_df()
    # Put the cluster value back to the dataset
    user_avg_rating_df['cluster'] = labels

    # Save it into a new csv file, you can view the result in this csv file
    user_avg_rating_df.sort_values(by=['cluster']).to_csv("./data/User_Cluster.csv")

    print("Cluster labels saved in ./data/User_Cluster.csv Successfully! ")

    return labels


if __name__ == "__main__":
    mergings = draw_user_dendrogram()
    HC_cluster_user(mergings,12.5)