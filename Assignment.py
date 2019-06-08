#!/usr/bin/env python
# coding: utf-8

# ## CAB420 Final Assignment
# ### Author: KA LONG LEE ( N9845097 )

# # Importing Library and Read Data from CSV files
# 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# Spliting data to trainning and testing set
from sklearn.model_selection import train_test_split

# Matplotlib setting 
get_ipython().run_line_magic('config', "InlineBackend.print_figure_kwargs = {'bbox_inches':None}")
get_ipython().run_line_magic('config', "InlineBackend.rc = {'font.size': 30, 'figure.figsize': (30.0, 20.0), 'figure.facecolor': (1, 1, 1, 0), 'figure.subplot.bottom': 0.125, 'figure.edgecolor': (1, 1, 1, 0), 'figure.dpi': 500}")


# ## Data Preprocessing

# In[2]:


from dataset import convert_dummy_movie_df, generate_rating_df_With_avg_rating 
from dataset import get_tags_df, generate_full_df, generate_user_avg_rating_df, generate_NB_df


# ## Movie Dataframe

# In[3]:


movie_df = convert_dummy_movie_df();
movie_df.head()


# ## Rating Dataframe

# In[4]:


rating_df = generate_rating_df_With_avg_rating();
rating_df.head()


# ## Tags Dataframe

# In[5]:


tags_df = get_tags_df()
tags_df.head()


# ## Full information Dataset (rating + tags + movies)

# In[6]:


full_rating_dataset = generate_full_df()
full_rating_dataset


# ## Random Forest Algorithm

# In[16]:


from sklearn.utils import shuffle
full_rating_dataset = shuffle(full_rating_dataset)

from RandomForest import Random_Forest_Prediction

X = full_rating_dataset.drop(columns=['rating'], axis=1).values
y = full_rating_dataset.loc[:,'rating'].astype(str).values

# Currently n_estimator is set to 100, it takes a long time to train the classifier
# If you are using a normal computer, you can set the n_estimator to 20 to make the trainning faster
# However, the accurarcy will reduce a little bit

Random_Forest_Prediction(X,y,100)


# ## Neural Network
# ### Neural Network is used to predict the user rating, please check neural_network.py which is saved in Rune - NN folder

# ## Naive Bayes Gaussian Models

# In[8]:


df = generate_NB_df()

df.head()


# In[9]:


from NaiveBayes import Naive_Bayes_Gaussian_Prediction
X = df.drop(columns=['rating'], axis=1).values
y = df['rating'].astype(str).values

Naive_Bayes_Gaussian_Prediction(X,y,0.2)


# # K-Means Algorithm Clustering Similar Movies

# In[10]:


# The dataset will use in this case
movie_df.iloc[:,1:-3].head()

movie_df


# In[11]:


from Kmeans import KMeans_SSE_Graphic, KMeans_Cluster

# X only contains movie genres values
X = movie_df.iloc[:,1:-3].values

KMeans_SSE_Graphic(X,50,"Movies")


# ## We Can see the result from the graph. The numbers of Inertias are dramatically decreased when k= 1 to k = 20, from 16000 to 6000. Then it becomes slow, from 6000 to 4000 when k = 21 to k=50. number of optimal cluster is K = 20 in this case
# 

# In[12]:


KMeans_Cluster(movie_df,X,20,"./data/KMeans_Movies_Cluster.csv")


# ## Hierarchy Clustering Similar Users
# 

# We need to create a new dataframe from the full rating dataset
# The new dataframe should contains all the average rating the user previous rating for all the genre

# In[13]:


get_ipython().run_line_magic('matplotlib', 'inline')
from Hierarchy_Clustering import HC_cluster_user, draw_user_dendrogram

mergings = draw_user_dendrogram()


# In[14]:


X = generate_user_avg_rating_df().iloc[:,1:].values

KMeans_SSE_Graphic(X,50,"Users")


# In[15]:


HC_cluster_user(mergings, 12.5)


# In[ ]:




