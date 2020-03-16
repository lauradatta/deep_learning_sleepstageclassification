# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 16:10:36 2020

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
    layer1 = Conv1D (filters = 64, kernel_size = 50, strides = 6, padding = 'same', activation = 'relu') (input_raw)
    layer1 = MaxPooling1D (pool_size = 8, strides = 8, padding = 'same') (layer1)
    layer1 = Dropout(0.5)(layer1)
    layer1 = Conv1D (128, 8, strides = 1, padding = "same", activation = 'relu') (layer1)
    layer1 = Conv1D (128, 8, strides = 1, padding = "same", activation = 'relu') (layer1)
    layer1 = Conv1D (128, 8, strides = 1, padding = "same", activation = 'relu') (layer1)
    layer1 = MaxPooling1D (pool_size = 4, strides = 4, padding = 'same') (layer1)
    layer1 = Flatten()(layer1)
    
    
    layer2 = Conv1D (filters = 64, kernel_size = 400, strides = 50, padding = 'same', activation = 'relu') (input_raw)
    layer2 = MaxPooling1D (pool_size = 4, strides = 4, padding = 'same') (layer2)
    layer2 = Dropout(0.5)(layer2)
    layer2 = Conv1D (128, 6, strides = 1, padding = "same", activation = 'relu') (layer2)
    layer2 = Conv1D (128, 6, strides = 1, padding = "same", activation = 'relu') (layer2)
    layer2 = Conv1D (128, 6, strides = 1, padding = "same", activation = 'relu') (layer2)
    layer2 = MaxPooling1D (pool_size = 4, strides = 4, padding = 'same') (layer2)
    layer2 = Flatten()(layer2)
    
    
    comb = concatenate([layer1, layer2])
    comb = LSTM(64, return_sequences = True) (comb)
    comb = LSTM (64)(comb)
    
    output_layer= Dense(6, activation = 'softmax')(comb)
    
    optimizer = Adam (lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, decay = 0, epsilon = (10 ** -8))
    model = Model (inputs = input_raw, outputs = output_layer)
    model.compile (loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    
    model.summary()
    
    return model

model = get_model()
model.summary()

#%%
hist = model.fit(raw_train, y_train_hot,
          validation_data = (raw_test, y_test_hot),
          epochs=20,
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


cat, test_acc = model.evaluate(raw_test, y_test_hot, batch_size=128)
print("accuracy score on test set is:{}".format(round(test_acc, 3)))
