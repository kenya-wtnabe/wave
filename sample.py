import sys
import math
import scipy as sc
import numpy as np
import pandas
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils

Datapath=''




model = Sequential([
        Dense(512, input_shape=(784,)),
        Activation('sigmoid'),
        Dense(10),
        Activation('softmax')
    ])

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=200, verbose=1, epochs=20, validation_split=0.1)

score = model.evaluate(X_test, y_test, verbose=1)
print('test accuracy : ', score[1])
