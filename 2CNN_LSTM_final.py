## import
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, LSTM, Dense, Input, Dropout,Flatten, \
    Concatenate, Reshape
from keras import optimizers
from keras import Model
import keras.backend as K
from keras.utils import to_categorical

#### preprocessing part ####

## set random seed for SMOTE
sm = SMOTE(random_state=42)

# Load data
with open ('data/Data_Raw_signals.pkl', 'rb') as f:
    raw = pickle.load (f)

with open ('data/Data_Spectrograms.pkl', 'rb') as f:
    spect = pickle.load (f)

#load test data with no labels
with open ('test_data/Test_Raw_signals_no_labels.pkl', 'rb') as f:
    raw_test = pickle.load (f)

with open ('test_data/Test_Spectrograms_no_labels.pkl', 'rb') as f:
    spec_test = pickle.load (f)

#### split labels from features
raw_train = np.array (raw [0])  # full data

spec_train = np.array (spect [0])  # full data
y_train = np.array (raw [1])  # labels

## remove noisy samples
raw_train.shape
elct1 = raw_train[:,0]
elct2 = raw_train[:,1]
too_high = []
getal = 2000
for i in range(elct1.shape[0]):
    temp = elct1[i]
    temp1 = elct2[i]
    numbermax = temp[np.argmax(temp)]
    numbermax1 = temp1[np.argmax (temp1)]
    numbermin = temp[np.argmin(temp)]
    numbermin1 = temp1[np.argmin (temp1)]
    if numbermax > getal or numbermin <(-getal) or numbermax1 > getal or numbermin1 <(-getal):
        too_high.append(i)

raw_train = np.delete(raw_train, too_high, axis = 0)
spec_train = np.delete(spec_train, too_high, axis = 0)
y_train = np.delete(y_train, too_high, axis = 0)

# hot encode the labels
y_train_hot = to_categorical (y_train, num_classes = 6)

## extract data for 1st and 2nd electrode - raw signals
train_raw_1 = raw_train[:, 0]  # first electrode
train_raw_2 = raw_train[:, 1]  # second electrode

test_raw_1 = raw_test[0][:, 0] # first electrode
test_raw_2 = raw_test[0][:, 1] # second electrode

## extract data for 1st and 2nd electrode - spectogram
train_spec_1 = spec_train [:, 0]  # first electrode
train_spec_2 = spec_train [:, 1]  # second electrode

test_spec_1 = spec_test[0][:, 0]  # first electrode
test_spec_2 = spec_test[0][:, 1]  # second electrode

# standardize raw signal 1
scaler_sig1 = StandardScaler()
train_raw_1 = scaler_sig1.fit_transform (train_raw_1)
test_raw_1 = scaler_sig1.transform(test_raw_1)

# standardize raw signal 2
scaler_sig2 = StandardScaler()
train_raw_2 = scaler_sig2.fit_transform (train_raw_2)
test_raw_2 = scaler_sig2.transform(test_raw_2)

## oversample the data with SMOTE
X_smote_raw, y_smote_raw = sm.fit_resample(train_raw_1, y_train_hot)
X_smote_raw2, y_smote_raw2 = sm.fit_resample(train_raw_2, y_train_hot)

#concatenate the raw signals and reshape to shape (15375, 3000, 2)
raw_train = np.stack ((X_smote_raw, X_smote_raw2), axis = 1)
raw_train = raw_train.reshape (raw_train.shape[0],raw_train.shape[2], raw_train.shape[1])

raw_test = np.stack ((test_raw_1, test_raw_2), axis = 1)
raw_test = raw_test.reshape (raw_test.shape[0],raw_test.shape[2], raw_test.shape[1])

## this creates 100 standardizers (for every 0.5 hz)
scalers1 = {}
for i in range(train_spec_1.shape[1]):
    scalers1[i] = StandardScaler()
    train_spec_1[:, i, :] = scalers1[i].fit_transform(train_spec_1[:, i, :])
    test_spec_1[:, i, :] = scalers1[i].transform(test_spec_1[:, i, :])

scalers2 = {}
for i in range(train_spec_2.shape[1]):
    scalers2[i] = StandardScaler()
    train_spec_2[:, i, :] = scalers2[i].fit_transform(train_spec_2[:, i, :])
    test_spec_2 [:, i, :] = scalers2 [i].transform (test_spec_2 [:, i, :])

## oversample the data with SMOTE
spec_train_1 = train_spec_1.reshape(train_spec_1.shape[0],3000)
spec_train_2 = train_spec_2.reshape(train_spec_2.shape[0],3000)

X_smote_spec1, y_smote_spec1 = sm.fit_resample(spec_train_1, y_train_hot)
X_smote_spec2, y_smote_spec2 = sm.fit_resample(spec_train_2, y_train_hot)

X_smote_spec1 = X_smote_spec1.reshape(X_smote_spec1.shape[0],100,30)
X_smote_spec2 = X_smote_spec2.reshape(X_smote_spec2.shape[0],100,30)

#concatenate the spectograms and reshape to shape (15375, 100, 30, 2,)
spec_train = np.stack((X_smote_spec1,X_smote_spec2), axis = 1)

spec_train = spec_train.reshape(spec_train.shape[0],
                                spec_train.shape[2],
                                spec_train.shape[3],
                                spec_train.shape[1])

spec_test = np.stack ((test_spec_1,test_spec_2), axis = 1)
spec_test = spec_test.reshape(spec_test.shape[0],
                              spec_test.shape[2],
                              spec_test.shape[3],
                              spec_test.shape[1])

## change y_train_hot to y_smote_raw
y_train_hot = y_smote_raw

#### build model ####
K.clear_session()
## first part of the model (raw signals as input)
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

## second part of the model (spectograms as input)
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

## concatenate the two part and reshape for the LSTM part
comb = Concatenate ()([layer1, layer2])
comb = Dropout(0.5)(comb)
lstm_input = Reshape ((1, K.int_shape(comb)[1])) (comb)

##LSTM part
lstm_part = LSTM (64, return_sequences = True) (lstm_input)
lstm_part = LSTM (32) (lstm_part)

## Dense layer and outputlayer
dense_part = Dense(32, activation = 'relu') (lstm_part)
output_layer = Dense (6, activation = 'softmax') (dense_part)

## create model and compile with Adam optimizer
model = Model (inputs = [input_raw,input_spec], outputs = output_layer)
optimizer = optimizers.Adam (lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, decay = 0, epsilon = (10 ** -8)) ## ff naar kijken nog
model.compile (loss = 'categorical_crossentropy',
                  optimizer = optimizer,
                  metrics = ['accuracy'])

## fit the model
model.fit([raw_train, spec_train], y_train_hot,
          epochs=200,
          batch_size=128,
          verbose =2)

## predict the labels of the test data
y_pred = model.predict([raw_test, spec_test])
y_pred = y_pred.argmax(axis=-1)
y_pred = y_pred.astype(int)

## save the labels as .txt file
with open('answer.txt', 'w') as f:
    for item in y_pred:
        f.write("%s\n" % item)
