
import numpy as np
import pandas as pd
import keras

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Embedding
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier

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
    model.fit(X_training, y_training, batch_size=1, epochs=20, verbose=0)
    score = [model.evaluate(X_test, y_test, verbose=0), model.evaluate(X_training, y_training, verbose=0)]

    return score

    # clf = KerasClassifier(model, verbose=0)
    # epochs = [10,50,100] # Amount of iterations over each traning set
    # batch_size = [1,2] # Different training sample sizes
    # optimizer = ['adam', 'sdg', 'rmsprop']
    # param_grid = dict(batch_size=batch_size, epochs=epochs, optimizer=optimizer)
    # grid = GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=-1)
    #
    # grid_result = grid.fit(X_training, y_training) # Training the classifier
    #
    # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    # means = grid_result.cv_results_['mean_test_score']
    # stds = grid_result.cv_results_['std_test_score']
    # params = grid_result.cv_results_['params']
    # for mean, stdev, param in zip(means, stds, params):
    #     print("%f (%f) with: %r" % (mean, stdev, param))





def calculate_scores(scores):
    sum_training_loss = 0
    sum_training_accuracy = 0
    sum_test_loss = 0
    sum_test_accuracy = 0
    best_training_accuracy = 0
    best_test_accuracy = 0

    for score in scores:
        if score[0][1] > best_training_accuracy:
            best_training_accuracy = score[0][1]
        if score[1][1] > best_test_accuracy:
            best_test_accuracy = score[1][1]

        sum_training_loss += score[0][0]
        sum_training_accuracy += score[0][1]
        sum_test_loss += score[1][0]
        sum_test_accuracy += score[1][1]

    score_size = len(scores)
    print('Best test accuracy = ', best_test_accuracy)
    print('Best training accuracy = ', best_training_accuracy)
    print('Average training loss = ', sum_training_loss/score_size)
    print('Average training accuracy = ', sum_training_accuracy/score_size)
    print('Average test loss = ', sum_test_loss/score_size)
    print('Average test accuracy = ', sum_test_accuracy/score_size)
    print(score[0])


if __name__ == "__main__":


    data = np.array(pd.read_csv('extracted_data.csv'))

    user_list = []
    data_list = []
    for entry in data:
        if entry[0] not in user_list:
            user_list.append(entry[0])
            data_list.append(get_user_array(data, entry[0]))

    user_list = np.array(user_list)
    data_list = np.array(data_list)
    scores = []
    for i in range(len(data_list)-500):
        X_training_user = np.array(data_list[i])
        y_training_user = X_training_user[:,[len(X_training_user[0])-1]]

        X_training, X_test, y_training, y_test = train_test_split(X_training_user[:,:-1], y_training_user , test_size=0.10)

        y_training = error_test_ratings(y_training)
        y_test = error_test_ratings(y_test)
        scores.append(build_NeuralNetwork_classifier(X_training, y_training, X_test, y_test))

    calculate_scores(scores)



    '''
    AVERAGE NUMBER OF RATINGS PR USER

    print(user_list[:5])
    sum = 0
    for x in data_list:
        sum += len(x)
    print((sum/len(data_list)))
    '''

    # X_training, X_test, y_training, y_test = train_test_split(data[:, 1:-1], data[:, [len(data[0])-1]], test_size=0.10)
    #
    # y_training = error_test_ratings(y_training)
    # y_test = error_test_ratings(y_test)
    #
    # build_NeuralNetwork_classifier(X_training, y_training, X_test, y_test)



    #print(data[0:5])
    #data, test_rating = fetch_data('extracted_data.csv')

    #nb.compute(train_input, train_output, X_test, y_test)
    #build_NeuralNetwork_classifier(train_input, train_output, X_test, y_test)
    # user = User(len(train_input[0]))
