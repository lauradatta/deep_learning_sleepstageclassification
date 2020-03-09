##% run preprocessing module
from preprocessing import X_train_spec, X_test_spec, y_train_hot, y_test_hot

#import packages
# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from keras import Model, layers
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Flatten, TimeDistributed, Input, Dropout, Reshape, Input, Dropout, Conv2D, MaxPooling2D
from keras.optimizers import Adam
import logging

# Remove some unwanted warnings
logging.getLogger('tensorflow').disabled = True

#%%


#%% define the model

def base_model():
    model = Sequential()
    model.add (Conv2D (8, 8, activation = 'relu', input_shape = (100,30,2)))
    model.add (MaxPooling2D (8))
    model.add (Conv2D (16, 8, activation = 'relu'))
    model.add (MaxPooling2D (8))
    model.add (Conv2D (32, (8), activation = 'relu'))
    model.add (MaxPooling2D (8))
    model.add (Flatten())
    model.add (LSTM (64, return_sequences = True))
    model.add (LSTM (64))
    model.add (Dense (6, activation = 'softmax'))

    optimizer = Adam (lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, decay = 0, epsilon = (10**-8))
    model.compile (loss = 'categorical_crossentropy',
                   optimizer = optimizer,
                   metrics = ['accuracy'])
    return model


model = base_model()
model.summary()

#%% # train the model

model.fit(X_train,y_train_hot,
          epochs=50,
          batch_size=128,
          verbose =2)

#%% evaluate the model

cat, test_acc = model.evaluate(X_test, y_test_hot, batch_size=128)
print("accuracy score on test set is:{}".format(round(test_acc, 3)))