#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 13:58:35 2019

@author: jenny
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# Spliting data to trainning and testing set
from sklearn.model_selection import train_test_split
# Fitting Multiple Linear Regression to the trainning set
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

# Import dataset
rating_dataset = pd.read_csv("ratings.csv")
movies_dataset = pd.read_csv("movies.csv")

# Set the index by movieId, This line of code only able to execute once
movies_dataset.set_index('movieId', inplace = True)

# Convert genres to dummy variable dataset
genresDummy = movies_dataset['genres'].str.get_dummies(sep='|')


# In[2]:


# Get all the genres values from the dataset
def get_all_genres():
    # a variable contains all the genre types
    genres = list()

    for row in movies_dataset.values:
        #  Sperating the genre by |
        Typestemp = row[1].split('|')
        #  Read all these type and put it to list
        for movietype in Typestemp:
            genres.append(movietype)
    return set(genres)


# In[3]:


# Add the dummy data back to the dataset
for genre in get_all_genres():
    movies_dataset[genre] = genresDummy[genre]
    
# Cannot use year as parameter because in movie 3xxxx there is a movie does not have years
# movies_dataset["year"] = movies_dataset["title"].str.extract(r"\(([0-9]+)\)").astype(dtype=np.int)

    
# Filtering duplicate values in the MovieGenre list
movies_dataset = movies_dataset.drop(columns=['genres', 'title'], axis=1)

movies_dataset


# In[4]:


full_rating_dataset = pd.merge(rating_dataset[["userId","movieId","rating"]], movies_dataset, on='movieId', how="left")
full_rating_dataset

# In[5]:
def get_avg_ratings(df): # df = full_rating_dataset
    
    num_users = df.userId.nunique() # 610
    num_genres = 20
    [m,n] = df.shape # (100836,23)
    avg_rating_matrix = np.zeros((num_users,num_genres))
    
    for user in range(1,num_users+1): # loop through users
        userId = df[df['userId'] == user] # dataframe selection for selected user
        genres = userId.iloc[:,3:23].values # numpy array of genre values
        genre_counter = np.zeros((1,num_genres))
        
        for row in range(userId.shape[0]): # loop through the user's rated movies
            idx = np.where(genres[row] == 1) # find index of genres
            avg_rating_matrix[user-1][idx[0]] +=df.iloc[row][2]
            genre_counter[0][idx[0]] +=1
        
        idx = np.where(genre_counter[0]!=0)
        for index in idx[0]:
            avg_rating_matrix[user-1][index] = avg_rating_matrix[user-1][index]/genre_counter[0][index]
                     
    return avg_rating_matrix
    

def count_genres(df):
    # Plot. Compare number of movies rated within each genre
    num_genres = 20
    [m,n] = df.shape # (100836,23)
    
    counter = np.zeros(num_genres)
    
    for genre in range(3,n):
        val = df.iloc[:,genre].values
        counter[genre-3] = len(np.where(val==1)[0])
    
    return counter

def avg_ratings_genres(df):
    # Plot. Compare the average rating over all users for each genre
    num_genres = 20
    num_users = df.userId.nunique() # 610
    [m,n] = df.shape # (100836,23)

    genre_count = count_genres(df)
    avg_matrix_users = get_avg_ratings(df)
    
    ratings = np.zeros(num_genres)
    
    for genre in range(num_genres):
        avg_rating = avg_matrix_users[:,genre].sum()
        ratings[genre] = avg_rating/num_users
    
    return ratings

def num_user_ratings(df):
    # Plot. Count number of each rating
    ratings_count = np.zeros(10)
    ratings = df.iloc[:,2].values
    
    values = np.linspace(0.5,5,10)
    idx = 0
    
    for i in values:
        ratings_count[idx] = len(np.where(ratings==i)[0])
        idx +=1
    
    return ratings_count

def make_plots(df,plot):
    
    lst_genres = list(df.columns[3:23])
    
    if plot==1:
        x = count_genres(df)
        num_bins = len(x)
        x = np.append(x,[0])
        plt.figure()
        n, bins, patches = plt.hist(np.linspace(0,20,21), weights = x, color='#0504aa', rwidth=0.6, bins = num_bins)
        plt.xlabel('Genre')
        plt.ylabel('Number of ratings')
        plt.title('Number of ratings made for each genre')
        plt.xticks(np.linspace(0.5,19.5,20), lst_genres, rotation='vertical')

        plt.savefig('Num_genre_ratings', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='png',
        transparent=False, bbox_inches='tight', pad_inches=0.1,
        frameon=None, metadata=None)
    
    elif plot==2:
        x = avg_ratings_genres(df)
        
        num_bins = len(x)
        x = np.append(x,[0])
        plt.figure()
        n, bins, patches = plt.hist(np.linspace(0,20,21), weights = x, color='#0504aa', rwidth=0.6, bins = num_bins)
        plt.xlabel('Genre')
        plt.ylabel('Rating')
        plt.title('Average rating for each genre')
        plt.xticks(np.linspace(0.5,19.5,20), lst_genres, rotation='vertical')
        
        plt.savefig('Average_genre_ratings', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='png',
        transparent=False, bbox_inches='tight', pad_inches=0.1,
        frameon=None, metadata=None)
        
    else:
        x = num_user_ratings(df)
        print(x)
        
        num_bins = len(np.linspace(0.5,5,10))
        #x = np.append(np.linspace(0.5,5,10),[0])
        plt.figure()
        n, bins, patches = plt.hist(np.linspace(0.5,5,10), weights = x, color='#0504aa', rwidth=0.6, bins = num_bins)
        plt.xlabel('User rating')
        plt.ylabel('Number of ratings')
        plt.title('All user ratings')
        plt.xticks(np.linspace(0.7,4.8,10), np.linspace(0.5,5,10))
        
        plt.savefig('User_ratings.png', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='png',
        transparent=False, bbox_inches='tight', pad_inches=0.1,
        frameon=None, metadata=None)

    

    
#a = get_avg_ratings(full_rating_dataset)

#b = count_genres(full_rating_dataset)

#c = avg_ratings_genres(full_rating_dataset)
    
#d = num_user_ratings(full_rating_dataset)

#e = get_all_genres()
    
#make_plots(full_rating_dataset,1)
#make_plots(full_rating_dataset,2)  
make_plots(full_rating_dataset,3)   
    
    
    
    
    
    
    
