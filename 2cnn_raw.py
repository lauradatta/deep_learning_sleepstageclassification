# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 16:10:36 2020

@author: Laura
"""


from keras.models import Sequential
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, LSTM, Dense, Input, Dropout,Flatten, Average,  concatenate, Concatenate, Reshape
from keras.optimizers import Adam, rmsprop
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
    comb = Dropout (0.5)(comb)
    comb = Reshape((1,2560))(comb)
    
    comb = LSTM(64, return_sequences = True) (comb)
    comb = LSTM (32)(comb)
    
    output_layer= Dense(6, activation = 'softmax')(comb)
    
    optimizer = rmsprop()
    model = Model (inputs = input_raw, outputs = output_layer)
    model.compile (loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    
    model.summary()
    
    return model

model = get_model()
model.summary()

#%%
# Define an input sequence and process it.
encoder_inputs = Input(shape=(maxlen_input, ctable.num_tokens))
encoder_rnn_layer = GRU(hidden_size, return_state=True)
# We discard the output of the layer and only keep the states.
_, encoder_state = encoder_rnn_layer(encoder_inputs)

### UPDATE CODE HERE ###
# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, ctable.num_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_rnn_layer = GRU(hidden_size, return_sequences=True, return_state=True)
decoder_outputs, _ = decoder_rnn_layer(decoder_inputs,
                                       initial_state=encoder_state)
decoder_dense = Dense(ctable.num_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
### END ###

# Define the model that will turn
# `encoder_inputs` & `decoder_inputs` into `decoder_outputs`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
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
