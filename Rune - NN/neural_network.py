
import numpy as np
import pandas as pd
import keras

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Embedding
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier

'''
'''
def get_user_array(data, user):
    data_list = []

    for entry in data:
        if entry[0] == user:
            data_list.append(entry[1:])

    return data_list

def error_test_ratings(y_validations):

    verification = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    validations_array = []
    for num in y_validations:
        this_validation = np.zeros(10)
        index = verification.index(num)
        this_validation[index] = 1
        validations_array.append(this_validation)

    return np.array(validations_array)


def generate_model():
    model = Sequential()
    model.add(Dense(20, input_dim=20, activation='sigmoid'))
    model.add(Dense(20, activation='sigmoid'))
    model.add(Dense(10, activation='sigmoid'))

    return model

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NeuralNetwork_classifier(X_training, y_training, X_test, y_test):

    model = generate_model()
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    # batch_size = 1 because we have such a small sample size to work with
    model.fit(X_training, y_training, batch_size=1, epochs=20, verbose=0)
    # scores = [result on test data, size of user data]
    score = [model.evaluate(X_test, y_test, verbose=0), len(X_test)]

    return score

'''
Creates a graph based on the output from our neural networks
@param scores: An array of [accuracy, mse] from the neural networks tested
@param data_list:
'''
def calculate_scores(scores):
    scores = np.array(scores)
    sum_test_loss = 0
    sum_test_accuracy = 0
    score_size = len(scores)

    test_loss = []
    test_accuracy = []
    test_sizes = []

    for i in range(len(scores)):
        sum_test_loss += scores[i][0][0]
        sum_test_accuracy += scores[i][0][1]
       # k_scores.append(scores[i][0][1]) #mean of 10 fold result
        test_loss.append(scores[i][0][0])
        test_accuracy.append(scores[i][0][1]*100) # in percent
        test_sizes.append(scores[i][1])

    linear_regressor = LinearRegression()  # create object for the class
    X = np.array(test_sizes).reshape(-1,1)
    Y = np.array(test_loss).reshape(-1,1)
    linear_regressor.fit(X, Y)  # perform linear regression
    Y_pred = linear_regressor.predict(X)  # make predictions

    plt.scatter(X, Y)
    plt.plot(X, Y_pred, color='red')
    plt.ylabel('Test mse')
    plt.xlabel('Data size')
    plt.title('Mean squared error')
    plt.savefig('foo5.png', bbox_inches='tight')
    plt.show()

    linear_regressor = LinearRegression()
    X = np.array(test_sizes).reshape(-1,1)
    Y = np.array(test_accuracy).reshape(-1,1)
    linear_regressor.fit(X, Y)  # perform linear regression
    Y_pred = linear_regressor.predict(X)  # make predictions

    plt.scatter(X, Y)
    plt.plot(X, Y_pred, color='red')
    plt.ylabel('Test Accuracy in %')
    plt.xlabel('Test data size')
    plt.title('Test Accuracy')
    plt.savefig('foo6.png', bbox_inches='tight')
    plt.show()

    print('Running training on 15% of the user\'s registered ratings')
    print('Average test loss = ', sum_test_loss/score_size)
    print('Average test accuracy = ', sum_test_accuracy/score_size)


if __name__ == "__main__":

    data = np.array(pd.read_csv('extracted_data.csv'))

    user_list = []
    data_list = []
    # this for loop seperates each user.
    for entry in data:
        if entry[0] not in user_list:
            user_list.append(entry[0])
            data_list.append(get_user_array(data, entry[0]))

    user_list = np.array(user_list)
    data_list = np.array(data_list)
    scores = []
    # training and finding results for each user
    for i in range(len(data_list)):
        X_training_user = np.array(data_list[i])
        y_training_user = X_training_user[:,[len(X_training_user[0])-1]]

        X_training, X_test, y_training, y_test = train_test_split(X_training_user[:,:-1], y_training_user , test_size=0.15)

        y_training = error_test_ratings(y_training)
        y_test = error_test_ratings(y_test)
        scores.append(build_NeuralNetwork_classifier(X_training, y_training, X_test, y_test))
        print('Finished user number : ', (i+1))

    calculate_scores(scores)
