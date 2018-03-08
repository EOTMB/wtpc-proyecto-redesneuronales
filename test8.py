# -*- coding: utf-8 -*-
"""
Created on Thu Mar 08 15:04:12 2018

@author: Yo
"""

import keras
import numpy as np



a = np.array([[1, 1], [0, 1], [1, 0], [0, 0]])
b = np.array([[0], [1], [1], [0]])

model = keras.Sequential()
model.add(keras.layers.Dense(2, activation = 'sigmoid', input_shape = (2, )))
model.add(keras.layers.Dense(1, activation = 'sigmoid'))

model.compile(optimizer = keras.optimizers.rmsprop(), metrics = ['accuracy'], loss = 'binary_crossentropy')
model.fit(x = a, y = b, epochs = 750)