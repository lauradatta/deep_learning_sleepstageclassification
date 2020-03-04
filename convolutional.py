##% run preprocessing module
from preprocessing import X_train, X_test, y_train_hot, y_test_hot

#import packages
# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from keras import Model, layers
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Flatten, TimeDistributed, Input, Dropout, Reshape
from keras.optimizers import Adam
import logging

# Remove some unwanted warnings
logging.getLogger('tensorflow').disabled = True

#%% Now seperapte CNN and lstm and add dropouts

def cnn_model():
    inp = Input(shape =(3000,2))
    layer=Conv1D (8, 8, activation = 'relu', padding = "valid")(inp)
    layer=MaxPooling1D (8)(layer)
    layer = Dropout(rate = 0.5)(layer)
    layer=Conv1D (16, 8, activation = 'relu')(layer)
    layer=MaxPooling1D(8)(layer)
    layer=Conv1D (32, (8), activation = 'relu')(layer)
    layer=MaxPooling1D(8)(layer)
    #flat = Flatten()(layer)
    out = Dense(64, activation = "relu")(layer)
    #model.add (LSTM (64, return_sequences = True))
    #model.add (LSTM (64))
    #model.add (Dense (6, activation = 'softmax'))
    
    cnn_model=Model(inputs = inp, outputs = out)    
    optimizer=Adam (lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, decay = 0, epsilon = (10**-8))
    cnn_model.compile (loss = 'categorical_crossentropy',
                   optimizer = optimizer,
                   metrics = ['accuracy'])
    cnn_model.summary()
    return cnn_model

#%% define lstm model

def lstm_model():
    lstm_inp = Input(shape=(3000,2))
    model_cnn = cnn_model()
    #for layer in model_cnn.layers:
    #    layer.trainable = False
    layer = (model_cnn)(lstm_inp)
    layer = LSTM (64, return_sequences = True)(layer)
    layer = Dropout(rate=0.5)(layer)
    layer = LSTM (64)(layer)
    out = Dense (6, activation = 'softmax')(layer)
    
    lstm_model = Model(lstm_inp, out)
    optimizer = Adam (lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, decay = 0, epsilon = (10**-8))
    lstm_model.compile (loss = 'categorical_crossentropy',
                   optimizer = optimizer,
                   metrics = ['accuracy'])
    
    return lstm_model

model_lstm = lstm_model()
model_lstm.summary()

#%%

X_train_resh = X_train.reshape(X_train.shape[0],3000,2,1)

#%% # train the model

model_lstm.fit(X_train,y_train_hot,
          epochs=30,
          batch_size=128,
          verbose =2)

#%% evaluate the model

cat, test_acc = model_lstm.evaluate(X_test, y_test_hot, batch_size=128)
print("accuracy score on test set is:{}".format(round(test_acc, 3)))

