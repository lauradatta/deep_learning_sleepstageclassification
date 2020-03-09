from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, LSTM, Dense, Input, Dropout,Flatten, Average
from keras.optimizers import Adam
from keras import Model
import numpy as np

from preprocess_train_test import raw_train, raw_test, spec_train, spec_test, y_train_hot

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
dense_1 = Dense(6, activation = 'softmax')(layer1)

input_spec = Input(shape =(spec_train.shape[1], spec_train.shape[2], spec_train.shape[3]))
layer2 = Conv2D(8,(3,3),activation = 'relu')(input_spec)
layer2 = MaxPooling2D(2,2)(layer2)
layer2 = Conv2D(16,(3,3),activation = 'relu')(layer2)
layer2 = MaxPooling2D(2,2)(layer2)
layer2 = Dropout(0.3)(layer2)
layer2 = Conv2D(32,(3,3),activation = 'relu')(layer2)
layer2 = Dropout(0.3)(layer2)
layer2 = Flatten()(layer2)
dense_2 = Dense(64, activation = 'softmax')(layer2)
dense_2 = Dense(6, activation = 'softmax')(layer2)

output_layer = Average()([dense_1, dense_2])

optimizer = Adam (lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, decay = 0, epsilon = (10 ** -8))
model = Model (inputs = [input_raw,input_spec], outputs = output_layer)
model.compile (loss = 'categorical_crossentropy',
                  optimizer = optimizer,
                  metrics = ['accuracy'])

model.summary()
hist = model.fit([raw_train, spec_train], y_train_hot,
          epochs=100,
          batch_size=128,
          verbose =2)

y_pred = model.predict([raw_test, spec_test])
y_pred = y_pred.argmax(axis=-1)
np.savetxt(y_pred)