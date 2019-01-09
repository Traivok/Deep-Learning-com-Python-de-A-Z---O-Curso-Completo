#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 19:12:49 2019

@author: ricardo
"""

import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')

##############################################################################
def createNeuralNetwork(optimizer, loss, kernel_initializer, activation, neurons):
    classifier = Sequential()
    
    # First Hidden Layer
    classifier.add(Dense(units = neurons, activation = activation,
                  kernel_initializer = kernel_initializer, input_dim = 30))
    classifier.add(Dropout(0.2))
    
    # Second Hidden Layer
    classifier.add(Dense(units = neurons, activation = activation,
                  kernel_initializer = kernel_initializer))
    classifier.add(Dropout(0.2))
    
    # Output
    classifier.add(Dense(units = 1, activation = 'sigmoid'))
        
    classifier.compile(optimizer = optimizer, loss = loss,
                      metrics = ['binary_accuracy'])
    return classifier
##############################################################################

classifier = KerasClassifier(build_fn = createNeuralNetwork,
                             epochs = 100, 
                             batch_size = 10)

params = {'batch_size': [10, 30], 
          'epochs': [50, 100],
          'optimizer': ['adam', 'sgd'],
          'loss': ['binary_crossentropy', 'hinge'],
          'kernel_initializer': ['random_uniform', 'normal'],
          'activation': ['relu', 'tanh'],
          'neurons': [16, 8, 32]}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = params,
                           scoring = 'accuracy',
                           cv = 5)

grid_search = grid_search.fit(previsores, classe)
best_params = grid_search.best_params_
best_accura = grid_search.best_score_