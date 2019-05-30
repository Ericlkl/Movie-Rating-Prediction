#!/usr/bin/env python
# coding: utf-8

# ## CAB420 Final Assignment
# 

# # Importing Library and Read Data from CSV files
# 

# In[1]:


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
movies_dataset = movies_dataset.drop(columns=['genres', 'title',"(no genres listed)","Western","IMAX"], axis=1)

movies_dataset


# In[4]:


full_rating_dataset = pd.merge(rating_dataset[["userId","movieId","rating"]], movies_dataset, on = 'movieId', how = "left")
full_rating_dataset


# ## Linear Regression for Predicting a user how many marks will he/she giving to a movie according to he/she previous rating to other movie and the others how they rate this movie
# 

# In[5]:


X = full_rating_dataset.drop(columns=['rating'], axis=1).values
y = full_rating_dataset.iloc[:,2].values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)


# In[6]:


y_pred


# In[7]:


y_test


# In[8]:


# The coefficients
print('Coefficients: \n', regressor.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))


# In[9]:


# Add 1 to Xtrain represent parameter0
X_train = np.append(arr = np.ones(( X_train.shape[0],1 )), values = X_train, axis = 1)


# In[10]:


X_opt = X_train[:,:]

# Backward Elimination
import statsmodels.api as smf

regressor_OLS = smf.OLS(endog= y_train, exog= X_opt).fit()
regressor_OLS.summary()


# ## Linear Regression for predicting a user will rate to a new movies according to the previous rating he gave to the other movies only

# In[11]:


# Randomly Generate a user for doing linear regression to predict what will he / she giving the rating on a movie
userID = random.randint(1, full_rating_dataset['userId'].max() + 1)

rating_df_for_one_user = full_rating_dataset.loc[(full_rating_dataset.userId == userID)]

X = rating_df_for_one_user.drop(columns=['rating','userId'], axis=1).values
y = rating_df_for_one_user.iloc[:,2].values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)
y_pred


# In[12]:


y_test


# In[13]:


# The coefficients
print('Coefficients: \n', regressor.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))


# In[14]:


# Add 1 to Xtrain represent parameter0
X_train = np.append(arr = np.ones(( X_train.shape[0],1 )), values = X_train, axis = 1)

X_opt = X_train[:,:]

# Backward Elimination
import statsmodels.api as smf

regressor_OLS = smf.OLS(endog= y_train, exog= X_opt).fit()
regressor_OLS.summary()


# In[15]:


rating_df_for_one_user.drop(columns=['rating','userId'], axis=1)


# In[ ]:




