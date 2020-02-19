##% run preprocessing module
from preprocessing import X_train, X_test, y_train_hot, y_test_hot

#import packages
# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from keras import Model
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Flatten, TimeDistributed, Input, Dropout
from keras.optimizers import Adam
import logging

# Remove some unwanted warnings
logging.getLogger('tensorflow').disabled = True

#%% define a base model

def base_model():
    model = Sequential()
    model.add (Conv1D (8, 8, activation = 'relu', input_shape = (X_train.shape[1], 2)))
    model.add (MaxPooling1D (8))
    model.add (Conv1D (16, 8, activation = 'relu'))
    model.add (MaxPooling1D (8))
    model.add (Conv1D (32, (8), activation = 'relu'))
    model.add (MaxPooling1D (8))
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



#%% Now seperapte CNN and lstm and add dropouts

def cnn_model():
    inp = Input(shape =(3000,))
    layer=Conv1D (8, 8, activation = 'relu', padding = "valid")(inp)
    layer=MaxPooling1D (8)(layer)
    layer = Dropout(rate = 0.5)(layer)
    layer=Conv1D (16, 8, activation = 'relu')(layer)
    layer=MaxPooling1D(8)(layer)
    layer=Conv1D (32, (8), activation = 'relu')(layer)
    layer=MaxPooling1D(8)(layer)
    out = Flatten()(layer)
    #model.add (LSTM (64, return_sequences = True))
    #model.add (LSTM (64))
    #model.add (Dense (6, activation = 'softmax'))
    
    base_model=Model(inputs = inp, outputs = out)    
    optimizer=Adam (lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, decay = 0, epsilon = (10**-8))
    base_model.compile (loss = 'categorical_crossentropy',
                   optimizer = optimizer,
                   metrics = ['accuracy'])
    return base_model


model_cnn = base_model()
model_cnn.summary()

#%% define lstm model

def lstm_model():
    inp = Input(shape=(None,3000,1))
    layer = TimeDistributed(model_cnn)(inp)
    layer = LSTM (64, return_sequences = True)(layer)
    layer = Dropout(rate=0.5)(layer)
    layer = LSTM (64)(layer)
    out = Dense (6, activation = 'softmax')(layer)
    
    lstm_model = Model(inp, out)
    optimizer = Adam (lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, decay = 0, epsilon = (10**-8))
    model.compile (loss = 'categorical_crossentropy',
                   optimizer = optimizer,
                   metrics = ['accuracy'])
    
    return model

model_lstm = lstm_model()
model_lstm.summary()
#%% # train the model

model_lstm.fit(X_train,y_train_hot,
          epochs=20,
          batch_size=128,
          verbose =2)

#%% evaluate the model

cat, test_acc = model_lstm.evaluate(X_test, y_test_hot, batch_size=128)
print("accuracy score on test set is:{}".format(round(test_acc, 3)))
