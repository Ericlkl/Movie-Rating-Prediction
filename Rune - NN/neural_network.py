
import numpy as np
import os
import random
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier


def generate_model():
    '''
    Generates a NN model based on layers and activations.
    @param
    @return
    model: The model of the Neural Network
    '''

    # model = Sequential()
    # model.add(Dense(21, input_dim=21, activation='sigmoid'))
    # model.add(Dense(10, activation='softmax'))
    # model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    input = Input((22,)) #supposing you have ten numeric values as input

    #here, SomeLayer() is defining a layer,
    #and calling it with (inp) produces the output tensor x
    layers = Dense(22, activation='sigmoid')(input)
    layers = Dense(10, activation='sigmoid')(layers)

    out1 = Dense(1, activation='sigmoid')(layers) # 0.5
    out2 = Dense(1, activation='sigmoid')(layers) # 1
    out3 = Dense(1, activation='sigmoid')(layers) # 1.5
    out4 = Dense(1, activation='sigmoid')(layers)
    out5 = Dense(1, activation='sigmoid')(layers)
    out6 = Dense(1, activation='sigmoid')(layers)
    out7 = Dense(1, activation='sigmoid')(layers)
    out8 = Dense(1, activation='sigmoid')(layers)
    out9 = Dense(1, activation='sigmoid')(layers)
    out10 = Dense(1, activation='sigmoid')(layers)


    #here, you define which path you will follow in the graph you've drawn with layers
    #notice the two outputs passed in a list, telling the model I want it to have two outputs.
    model = Model(input, [out1,out2,out3,out4,out5,out6,out7,out8,out9,out10])

    return model

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NeuralNetwork_classifier(X_training, y_training, X_test, y_test):
    '''
    Build a Neural Network with two dense hidden layers classifier
    based on the training set X_training, y_training.
    Use the Keras functions from the Tensorflow library
    @param
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]
    @return
	clf : the classifier built in this function
    '''

    #here, you define which path you will follow in the graph you've drawn with layers
    #notice the two outputs passed in a list, telling the model I want it to have two outputs.
    model = generate_model()
    model.compile(optimizer = 'adam', loss = 'mse', matrix=['accuracy'])

    model.fit(X_training, [out1,out2,out3,out4,out5,out6,out7,out8,out9,out10], epochs=20)
    # First finding the best hyperparameter

    # clf = KerasClassifier(build_fn=generate_model, verbose=0)
    # epochs = [20] # Amount of iterations over each traning set
    # batch_size = [20] # Different training sample sizes
    # param_grid = dict(batch_size=batch_size, epochs=epochs)
    # grid = GridSearchCV(estimator=clf, cv=10, param_grid=param_grid, n_jobs=-1)

    # grid_result = grid.fit(X_training, y_training) # Training the classifier
    #
    # print(grid_result.cv_results_)
    #
    #
    # scores = grid_result.cv_results_['mean_test_score']
    # scores_std = grid_result.cv_results_['std_test_score']
    # std_error = scores_std / np.sqrt(10)
    #
    # pred_train = grid_result.predict(X_training)
    # train_clf_errors = np.sum(y_training!=pred_train)
    # train_mse = mean_squared_error(pred_train,y_training)
    #
    # pred_test = grid_result.predict(X_test)
    # test_clf_errors = np.sum(y_test!=pred_test)
    # test_mse = mean_squared_error(pred_test,y_test)
    #
    # print('Number of errors on training data: ', train_clf_errors, '\nMSE for training data', train_mse)
    # print('Number of errors on test data: ', test_clf_errors, '\nMSE for test data', test_mse)
    #print('The best choice of ' + x_label + ': ' + str(clf.best_params_[tuned_param]),'\n')

    # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    # means = grid_result.cv_results_['mean_test_score']
    # stds = grid_result.cv_results_['std_test_score']
    # params = grid_result.cv_results_['params']
    # for mean, stdev, param in zip(means, stds, params):
    #     print("%f (%f) with: %r" % (mean, stdev, param))

    return clf

if __name__ == "__main__":

    #neural_network = NeuralNetwork()
    #data, results = extract_csv('data/movies.csv', 'data/ratings.csv')
    #create_file(data, results, 'extracted_data.csv')
    data = np.array(pd.read_csv('extracted_data.csv'))
    data = data[1:]
    test_rating = data[:, [21]]

    #print(data[0:5])
    #data, test_rating = fetch_data('extracted_data.csv')
    train_input, X_test, train_output, y_test = train_test_split(data, test_rating, test_size=0.20)
    #nb.compute(train_input, train_output, X_test, y_test)
    build_NeuralNetwork_classifier(train_input, train_output, X_test, y_test)
    # user = User(len(train_input[0]))
