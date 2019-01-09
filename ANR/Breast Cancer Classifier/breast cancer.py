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

def createNeuralNetwork():
    classifier = Sequential()
    classifier.add(Dense(units = 8, activation = 'relu',
                  kernel_initializer = 'normal', input_dim = 30))
    classifier.add(Dropout(0.2))  
    classifier.add(Dense(units = 8, activation = 'relu',
                  kernel_initializer = 'normal'))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))
    
    optimizer = keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.3)
    
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy',
                      metrics = ['binary_accuracy'])
    return classifier


previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')

classifier = KerasClassifier(build_fn = createNeuralNetwork,
                             epochs = 100, 
                             batch_size = 10)
results = cross_val_score(estimator = classifier,
                          X = previsores, y = classe,
                          cv = 10, scoring = 'accuracy')

mean = results.mean()
sdev = results.std()