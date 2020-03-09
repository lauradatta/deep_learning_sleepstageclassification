## run preprocessing module
from preprocessing import X_train_1, X_test_1,X_train_2, X_test_2,\
    y_train_hot, y_test_hot

from keras import Model
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Flatten, TimeDistributed, \
    Input, Dropout, concatenate, RepeatVector, dot
from keras.optimizers import Adam
from keras.utils import plot_model


import logging

## reshape train and test sets
X_train_1 = X_train_1.reshape(X_train_1.shape[0], X_train_1.shape[1], 1)
X_train_2 = X_train_2.reshape(X_train_2.shape[0], X_train_2.shape[1], 1)
X_test_1 = X_test_1.reshape(X_test_1.shape[0], X_test_1.shape[1], 1)
X_test_2 = X_test_2.reshape(X_test_2.shape[0], X_test_2.shape[1], 1)

#part 1 of CNN:
input_signal1 = Input(shape =(3000, 1))
layer1 = Conv1D (8, 8, activation = 'relu', padding = "valid") (input_signal1)
layer1 = MaxPooling1D (8) (layer1)
layer1 = Dropout (rate = 0.5) (layer1)
layer1 = Conv1D (16, 8, activation = 'relu') (layer1)
layer1 = MaxPooling1D (8) (layer1)
layer1 = Conv1D (32, (8), activation = 'relu') (layer1)
layer1 = MaxPooling1D (8) (layer1)
cnn_part1 = Flatten () (layer1)

#part 2
input_signal2 = Input(shape =(3000, 1))
layer2 = Conv1D (8, 8, activation = 'relu', padding = "valid") (input_signal2)
layer2 = MaxPooling1D (8) (layer2)
layer2 = Dropout (rate = 0.5) (layer2)
layer2 = Conv1D (16, 8, activation = 'relu') (layer2)
layer2 = MaxPooling1D (8) (layer2)
layer2 = Conv1D (32, (8), activation = 'relu') (layer2)
layer2 = MaxPooling1D (8) (layer2)
cnn_part2 = Flatten () (layer2)

## concatenate the two parts

x = concatenate([cnn_part1, cnn_part2])

## I have used a repeater for now because LSTM won't accept the output of layer x
x2 = RepeatVector(2) (x)

## LSTM layers
LSTM_1 = LSTM (256, return_sequences = True)(x2)
LSTM_2 = LSTM (64)(LSTM_1)

## output layer
dense_1 = Dense (6, activation = 'softmax')(LSTM_2)

optimizer = Adam (lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, decay = 0, epsilon = (10 ** -8))
model = Model (inputs =[input_signal1, input_signal2], outputs = dense_1)
model.compile (loss = 'categorical_crossentropy',
                  optimizer = optimizer,
                  metrics = ['accuracy'])

## fitting the model
model.fit([X_train_1, X_train_2], y_train_hot,
          epochs=100,
          batch_size=128,
          verbose =2)

## testing the model
cat, test_acc = model.evaluate([X_test_1, X_test_2], y_test_hot, batch_size=128)
print("accuracy score on test set is:{}".format(round(test_acc, 3)))


## for this part you have to install Graphviz2

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

## create image of the model
plot_model(model, to_file='cnn_model1.png', show_shapes = True)
