import numpy as np
import pandas as pd

def convert_dummy_movie_df():
    movies_dataset = pd.read_csv("./data/movies.csv")
                             
    # Set the index by movieId, This line of code only able to execute once
    movies_dataset.set_index('movieId', inplace = True)
                             
    # Convert genres to dummy variable dataset
    # movies_genres_dummy <- a dataframe contains the genres values for all the movies and using movieId as index
    movies_genres_dummy = movies_dataset['genres'].str.get_dummies(sep='|')
    # Remove (no genres listed) from dummy because all 0 can represent (no genres listed)
    movies_genres_dummy = movies_genres_dummy.drop(columns=["(no genres listed)"], axis=1)

    # Pre Processing for KMeans Algorithm finding the similar movies
    movies_dataset = movies_dataset.drop(columns=["genres"], axis=1)
    movies_dataset = pd.merge(movies_dataset, movies_genres_dummy, on = 'movieId', how = "left")
                             
    return movies_dataset.drop(columns=["title"], axis=1)

def generate_rating_df_With_avg_rating():
    # Import dataset
    rating_dataset = pd.read_csv("./data/ratings.csv")
    
    user_avg_rating = rating_dataset.groupby('userId')['rating'].mean().to_frame()
    user_avg_rating = user_avg_rating.rename(columns={"rating": "user_avg_rating"})

    movie_avg_rating = rating_dataset.groupby('movieId')['rating'].mean().to_frame()
    movie_avg_rating = movie_avg_rating.rename(columns={"rating": "movie_avg_rating"})

    rating_dataset = pd.merge(rating_dataset, user_avg_rating, on = 'userId', how = "left")
    rating_dataset = pd.merge(rating_dataset, movie_avg_rating, on = 'movieId', how = "left")

    return rating_dataset.drop(columns=["timestamp"], axis=1)

def get_movie_genres():
    return convert_dummy_movie_df().columns

def get_tags_df():
    tags_dataset = pd.read_csv("./data/tags.csv")
    tags_dataset['tag'] = tags_dataset['tag'].str.lower()
    # Currently we disable the userId
    return pd.get_dummies(tags_dataset).drop(columns=["timestamp", "userId"],axis=1)

def generate_full_df():
    # Import dataset
    rating_df = pd.read_csv("./data/ratings.csv")
    
    movie_df = convert_dummy_movie_df()
    tags_df = get_tags_df()
    # Merge Three dataset, tags + movie + rating, to the complete dataset
    full_rating_dataset = pd.merge(rating_df, movie_df, on = 'movieId', how = "left")

    # Merge user average rating for each genre to the dataframe    
    for genre in get_movie_genres():
        user_avg_genre_rating = full_rating_dataset.loc[(full_rating_dataset[genre] == 1)].groupby('userId')['rating'].mean().to_frame()
        user_avg_genre_rating = user_avg_genre_rating.rename(columns={"rating": "user_avg_"+ genre +"_rating"})
        full_rating_dataset = pd.merge(full_rating_dataset, user_avg_genre_rating, on = 'userId', how = "left")

    # return the non zero result dataframe
    return pd.merge(full_rating_dataset, tags_df, on = 'movieId', how = "left").fillna(0)

def generate_user_avg_rating_df():
    full_rating_dataset = generate_full_df()
    #genres_list includes ['Action', 'Adventure', 'Animation', 'Children', etc...]
    genres_list = get_movie_genres().values

    # Columns name variable for creating the new dataframe
    columns_name = ["userId"]
    # Create Columns name array
    for genres in genres_list:
        columns_name.append("AVG_" + genres + "_Rating")

    # initialize the dataframe     
    df = pd.DataFrame(columns=columns_name)

    for userId in range(1,full_rating_dataset['userId'].max() + 1):
        row_values = { 'userId': userId }
        for idx , genre in enumerate(genres_list):
            # Getting the user Dataframe for calculating the avg value for this genre
            user_df = full_rating_dataset.loc[(full_rating_dataset["userId"] == userId) & (full_rating_dataset[genre] == 1) ]
            # calculating the avg value for this genre
            avg = user_df["rating"].mean()

            row_values.update( { columns_name[idx + 1] : avg} )
            
        df = df.append( pd.Series(row_values), ignore_index = True )
    df[['userId']] = df[['userId']].astype(int)
    return df.fillna(0)