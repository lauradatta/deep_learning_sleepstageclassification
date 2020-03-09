# Import packages
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

# Load data (is data in data folder)
with open ('data/Data_Raw_signals.pkl', 'rb') as f:
    raw = pickle.load (f)

with open ('data/Data_Spectrograms.pkl', 'rb') as f:
    spect = pickle.load (f)

#### split labels from features
signal_raw = np.array (raw [0])  # full data
label_raw = np.array (raw [1])  # labels

signal_spec = np.array (spect [0])  # full data
label_spec = np.array (spect [1])  # labels

## extract data for 1st and 2nd electrode - spectogram
signal_spec_1 = signal_spec [:, 0]  # first electrode
signal_spec_2 = signal_spec [:, 1]  # second electrode

## extract data for 1st and 2nd electrode - raw signals
signal_raw_1 = signal_raw [:, 0]  # first electrode
signal_raw_2 = signal_raw [:, 1]  # second electrode

## this creates 100 standardizers (for every 0.5 hz) <-- Maybe we have to standardize the other way around.
scalers1 = {}
for i in range(signal_spec_1.shape[1]):
    scalers1[i] = StandardScaler()
    signal_spec_1[:, i, :] = scalers1[i].fit_transform(signal_spec_1[:, i, :])

scalers2 = {}
for i in range(signal_spec_2.shape[1]):
    scalers2[i] = StandardScaler()
    signal_spec_2[:, i, :] = scalers2[i].fit_transform(signal_spec_2[:, i, :])

#concatenate the spectograms and reshape to shape (15375, 100, 30, 2,)
signal_spec_st = np.stack ((signal_spec_1,signal_spec_2), axis = 1)

signal_spec_st = signal_spec_st.reshape(signal_spec_st.shape[0],
                                        signal_spec_st.shape[2],
                                        signal_spec_st.shape[3],
                                        signal_spec_st.shape[1])

# standardize raw signal 1
scaler_sig1 = StandardScaler ()
signal_raw_1_st = scaler_sig1.fit_transform (signal_raw_1)

# standardize raw signal 2
scaler_sig2 = StandardScaler ()
signal_raw_2_st = scaler_sig2.fit_transform (signal_raw_2)

#concatenate the raw signals and reshape to shape (15375, 3000, 2)
signal_raw_st = np.stack ((signal_raw_1_st, signal_raw_2_st), axis = 1)
signal_raw_st = signal_raw_st.reshape (signal_raw_st.shape[0],signal_raw_st.shape[2], signal_raw_st.shape[1])

#split data into train and test set
raw_train, raw_test, y_train, y_test = train_test_split (signal_raw_st, label_raw,
                                                     test_size = 0.2, random_state = 123)

spec_train, spec_test, y_train_s, y_test_s = train_test_split (signal_spec_st, label_spec,
                                                     test_size = 0.2, random_state = 123)

# hot label the labels
y_train_hot = to_categorical (y_train, num_classes = 6)
y_test_hot = to_categorical (y_test, num_classes = 6)

