# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 13:19:21 2020

@author: Laura
"""

from keras.models import Sequential
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, LSTM, Dense, Input, Dropout,Flatten, Average,  concatenate, Concatenate
from keras.optimizers import Adam
from keras import Model
import logging

from preprocess_small import raw_train, raw_test, spec_train, spec_test, y_train_hot, y_test_hot

#%%

def get_model():
    input_raw = Input(shape =(raw_train.shape[1], 2))
    layer1 = Conv1D (8, 8, activation = 'relu') (input_raw)
    layer1 = MaxPooling1D (8) (layer1)
    layer1 = Conv1D (16, 8, activation = 'relu') (layer1)
    layer1 = MaxPooling1D (8) (layer1)
    layer1 = Dropout(0.3)(layer1)
    layer1 = Conv1D (32, 8, activation = 'relu') (layer1)
    layer1 = MaxPooling1D (8) (layer1)
    layer1 = LSTM(64, return_sequences = True) (layer1)
    layer1 = LSTM(64) (layer1)
    dense1 = Dense(64, activation = 'relu')(layer1)
    
    
    
    input_spec = Input(shape =(spec_train.shape[1], spec_train.shape[2], spec_train.shape[3]))
    layer2 = Conv2D(8,(3,3),activation = 'relu')(input_spec)
    layer2 = MaxPooling2D(2,2)(layer2)
    layer2 = Conv2D(16,(3,3),activation = 'relu')(layer2)
    layer2 = MaxPooling2D(2,2)(layer2)
    layer2 = Dropout(0.3)(layer2)
    layer2 = Conv2D(32,(3,3),activation = 'relu')(layer2)
    layer2 = Dropout(0.3)(layer2)
    layer2 = Flatten()(layer2)
    dense2 = Dense(64, activation = 'relu')(layer2)
    
    comb = concatenate([dense1, dense2])
    
    comb = Dense(32, activation = 'relu')(comb)
    output_layer= Dense(6, activation = 'softmax')(comb)
    
    optimizer = Adam (lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, decay = 0, epsilon = (10 ** -8))
    model = Model (inputs = [input_raw,input_spec], outputs = output_layer)
    model.compile (loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    
    model.summary()
    
    return model

#%%

model = get_model()

#%%

hist = model.fit([raw_train, spec_train], y_train_hot,
          validation_data = ([raw_test, spec_test], y_test_hot),
          epochs=2,
          batch_size=128,
          verbose =2)

history_dict = hist.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
accuracy = history_dict['accuracy']
val_accuracy = history_dict['val_accuracy']

from plot_history import plot_history

plot_history([loss, val_loss],["loss","val_loss"], "checking it out", "loss")

plot_history([accuracy, val_accuracy],["accuracy","val_accuracy"], "checking it out", "accuracy")


cat, test_acc = model.evaluate([raw_test, spec_test], y_test_hot, batch_size=128)
print("accuracy score on test set is:{}".format(round(test_acc, 3)))