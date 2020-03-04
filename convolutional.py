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
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Flatten, TimeDistributed, Input, Dropout, Reshape, Input, Dropout, Conv1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, concatenate, SpatialDropout1D, Bidirectional
from keras.optimizers import Adam
import logging

# Remove some unwanted warnings
logging.getLogger('tensorflow').disabled = True

#%% Now seperapte CNN and lstm and add dropouts

def cnn_lstm_model():
    inp = Input(shape =(3000,2))
    layer=Conv1D (8, 8, activation = 'relu', padding = "valid")(inp)
    layer=MaxPooling1D (8)(layer)
    layer = Dropout(rate = 0.5)(layer)
    layer=Conv1D (16, 8, activation = 'relu')(layer)
    layer=MaxPooling1D(8)(layer)
    layer=Conv1D (32, (8), activation = 'relu')(layer)
    layer=MaxPooling1D(8)(layer)
    #flat = Flatten()(layer)
    
    layer = LSTM (64, return_sequences = True)(layer)
    layer = Dropout(rate=0.5)(layer)
    layer = LSTM (64)(layer)
    out = Dense (6, activation = 'softmax')(layer)
    
    lstm_model = Model(inp, out)
    optimizer = Adam (lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, decay = 0, epsilon = (10**-8))
    lstm_model.compile (loss = 'categorical_crossentropy',
                   optimizer = optimizer,
                   metrics = ['accuracy'])
    
    return lstm_model

model_cnn_lstm = cnn_lstm_model()
model_cnn_lstm.summary()


#%% # train the model

model_lstm.fit(X_train,y_train_hot,
          epochs=50,
          batch_size=128,
          verbose = 2)

#%% evaluate the model

cat, test_acc = model_lstm.evaluate(X_test, y_test_hot, batch_size=128)
print("accuracy score on test set is:{}".format(round(test_acc, 3)))

#%%
def cnn_model2():
    inp = Input(shape=(3000, 2))
    img_1 = Conv1D(16, kernel_size=5, activation='relu', padding="valid")(inp)
    img_1 = Conv1D(16, kernel_size=5, activation='relu', padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = SpatialDropout1D(rate=0.01)(img_1)
    img_1 = Conv1D(32, kernel_size=3, activation='relu', padding="valid")(img_1)
    img_1 = Conv1D(32, kernel_size=3, activation='relu', padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = SpatialDropout1D(rate=0.01)(img_1)
    img_1 = Conv1D(32, kernel_size=3, activation='relu', padding="valid")(img_1)
    img_1 = Conv1D(32, kernel_size=3, activation='relu', padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = SpatialDropout1D(rate=0.01)(img_1)
    img_1 = Conv1D(256, kernel_size=3, activation='relu', padding="valid")(img_1)
    img_1 = Conv1D(256, kernel_size=3, activation='relu', padding="valid")(img_1)
    #img_1 = GlobalMaxPool1D()(img_1)
    #img_1 = Dropout(rate=0.01)(img_1)
    
    dense_1 = Dropout(0.01)(Dense(64, activation='relu', name="dense_1")(img_1))

    base_model=Model(inputs = inp, outputs = dense_1)    
    optimizer=Adam (lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, decay = 0, epsilon = (10**-8))
    base_model.compile (loss = 'categorical_crossentropy',
                   optimizer = optimizer,
                   metrics = ['accuracy'])
    base_model.summary()
    return base_model

def lstm_model2():
    lstm_inp = Input(shape=(3000,2))
    model_cnn = cnn_model2()
    #nclass = 6
    #for layer in model_cnn.layers:
    #    layer.trainable = False
    layer = (model_cnn)(lstm_inp)
    layer = LSTM(100, return_sequences = True)(layer)
    encoded_sequence = Dropout(rate=0.5)(layer)
    encoded_sequence = LSTM(100, return_sequences = False)(encoded_sequence)
    #out = TimeDistributed(Dense(nclass, activation="softmax"))(encoded_sequence)
    #out = Conv1D(nclass, kernel_size=1, activation="softmax", padding="same")(encoded_sequence)
    out = Dense (6, activation = 'softmax')(encoded_sequence)
    
    lstm_model = Model(lstm_inp, out)
    optimizer = Adam (lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, decay = 0, epsilon = (10**-8))
    lstm_model.compile (loss = 'categorical_crossentropy',
                   optimizer = optimizer,
                   metrics = ['accuracy'])
    
    return lstm_model

model_lstm2 = lstm_model()
model_lstm2.summary()


#%% # train the model

model_lstm2.fit(X_train,y_train_hot,
          epochs=50,
          batch_size=100,
          verbose = 2)
#210
#%% evaluate the model

cat, test_acc = model_lstm2.evaluate(X_test, y_test_hot, batch_size=128)
print("accuracy score on test set is:{}".format(round(test_acc, 3)))

