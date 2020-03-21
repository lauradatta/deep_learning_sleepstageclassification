## import
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, LSTM, Dense, Input, Dropout,Flatten, \
    Concatenate, Reshape
from keras import optimizers
from keras import Model
import keras.backend as K

from train_test_dim_clean_remove import raw_train, raw_test, spec_train, spec_test, y_train_hot

input_raw = Input(shape =(raw_train.shape[1], 2))
layer1 = Conv1D (8, 8, activation = 'relu') (input_raw)
layer1 = MaxPooling1D (8) (layer1)
layer1 = Conv1D (16, 8, activation = 'relu') (layer1)
layer1 = MaxPooling1D (8) (layer1)
layer1 = Dropout(0.3)(layer1)
layer1 = Conv1D (32, 8, activation = 'relu') (layer1)
layer1 = MaxPooling1D (8) (layer1)
layer1 = Dropout(0.3)(layer1)
layer1 = Flatten()(layer1)

input_spec = Input(shape =(spec_train.shape[1], spec_train.shape[2], spec_train.shape[3]))
layer2 = Conv2D(8,(3,3),activation = 'relu')(input_spec)
layer2 = MaxPooling2D(2,2)(layer2)
layer2 = Conv2D(16,(3,3),activation = 'relu')(layer2)
layer2 = MaxPooling2D(2,2)(layer2)
layer2 = Dropout(0.5)(layer2)
layer2 = Conv2D(32,(3,3),activation = 'relu')(layer2)
layer2 = MaxPooling2D(2,2)(layer2)
layer2 = Dropout(0.5)(layer2)
layer2 = Flatten()(layer2)

comb = Concatenate ()([layer1, layer2])
comb = Dropout(0.5)(comb)
comb = Reshape ((1, K.int_shape(comb)[1])) (comb)

comb = LSTM (64, return_sequences = True) (comb)
comb = LSTM (32) (comb)

comb = Dense(32, activation = 'relu') (comb)

output_layer = Dense (6, activation = 'softmax') (comb)

optimizer = optimizers.Adam (lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, decay = 0, epsilon = (10 ** -8))

model = Model (inputs = [input_raw,input_spec], outputs = output_layer)
model.compile (loss = 'categorical_crossentropy',
                  optimizer = optimizer,
                  metrics = ['accuracy'])

model.summary()
hist = model.fit([raw_train, spec_train], y_train_hot,
          epochs=200,
          batch_size=128,
          verbose =2)

y_pred = model.predict([raw_test, spec_test])
y_pred = y_pred.argmax(axis=-1)
y_pred = y_pred.astype(int)

with open('answer.txt', 'w') as f:
    for item in y_pred:
        f.write("%s\n" % item)
