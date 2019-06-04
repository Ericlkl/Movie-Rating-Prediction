import csv
import numpy as np

def find_genres(movie_list):
    genres = []
    for movie in movie_list:
        for genre in movie[1]:
            if genre not in genres:
                genres.append(genre)
    print(genres)
    return genres

def create_input_data(movie_list, rating_list):
    genres = find_genres(movie_list)
    input = []

    for movie in movie_list:
        zeros = np.zeros(len(genres))
        for genre in movie[1]:
            zeros[genres.index(genre)] = 1

        for r in rating_list:
            #input.append(rating_list[r])
            if movie[0] == r[1]:
                temp_front = []
                temp_front.append(r[0])
                temp_front.extend(zeros)
                temp_front.append(float(r[2]))
                input.append(temp_front)

    return input

def split_movies(line):
    escape_sign = False
    elem_list = []
    elem_start = 0
    for i in range(len(line)-1):
        if line[i] == '\"':
            escape_sign = True if not escape_sign else False
        if line[i] == ',' and not escape_sign:
            elem_list.append(line[elem_start:i])
            elem_start = i+1
    elem_list.append(line[elem_start:-1].split('|')) # creates list with | as seperator, and removes new line
    return np.array(elem_list)[[0,2]] # Removing movie title


def extract_csv(file_path_movies, file_path_ratings):
    # Opening the files
    movies_file = open(file_path_movies, encoding='utf8')
    ratings_file = open(file_path_ratings, encoding='utf8')
    # Skipping header line
    print(movies_file.readline())
    print(ratings_file.readline())
    # initiating local variables
    movies_data = []
    ratings_data = []
    test_data = []
    # Getting data from file
    for line in movies_file:
        movies_data.append(split_movies(line))
    for line in ratings_file:
        ratings_data.append(line.split(','))
    # Closing files
    movies_file.close()
    ratings_file.close()

    input_data = np.array(create_input_data(movies_data, ratings_data))
    results = input_data[:, -1]
    print(results[0:10])
    print(input_data[0:10])

    numpy_movies = np.array(movies_data)
    numpy_ratings = np.array(ratings_data)

    return input_data[:,0:len(input_data[0])-1], results

def create_file(input_data, rating_data, file_name):
    f = open(file_name, 'w') # Creates the file if not found
    f.write('UserID,Adventure,Animation,Children,Comedy,Fantasy,Romance,Drama,Action,Crime, \
    Thriller,Horror,Mystery,Sci-fi,War,Musical,Documentary,IMAX,Western,Film-Noir,(no genres listed),Rating')
    for i in range(len(input_data)):
        temp_string = ','.join(input_data[i])
        f.write(','.join([temp_string, rating_data[i]]))
        f.write('\n')

    f.close()
