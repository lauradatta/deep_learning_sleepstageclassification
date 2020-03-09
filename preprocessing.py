# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import acf
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

# %%  Load data

# working directory is set at level of project

# Load data (is data in data folder)
with open ('data/Data_Raw_signals.pkl', 'rb') as f:
    raw = pickle.load (f)

with open ('data/Data_Spectrograms.pkl', 'rb') as f:
    spect = pickle.load (f)

# %%  Put data in good format

#### Raw signals
signal_raw = np.array (raw [0])  # full data
label_raw = np.array (raw [1])  # labels

## extract data for 1st and 2nd electrode
signal_raw_1 = signal_raw [:, 0]  # first electrode
signal_raw_2 = signal_raw [:, 1]  # second electrode

#### spectograms ###
signal_spect = np.array (spect [0])  # full data
label_spect = np.array (spect [1])  # labels

## extract data for 1st and 2nd electrode
signal_spec_1 = signal_spect [:, 0]  # first electrode
signal_spec_2 = signal_spect [:, 1]  # second electrode

# %% Visualise the data for the first person

# Raw signals
plt.plot (signal_raw_1 [0, :])  # electrode 1 for first person
plt.plot (signal_raw_2 [0, :])  # electrode 2 for first person

# Spectograms
plt.plot(signal_spec_1 [0, :])  # electrode 1 for first person
plt.plot (signal_spec_2 [0, :])  # electrode 2 for first person

##  checking autocorrelation ###
acf (signal_raw_1 [0, :])
acf (signal_raw_2 [0, :])
autocorrelation_plot (signal_raw_1 [0, :])
autocorrelation_plot (signal_raw_2 [0, :])

# %% RAW SIGNAL

#checking basics

# shapes

signal_raw.shape
signal_raw_1.shape
signal_raw_2.shape
# We have 3000 data points on for 2 electrodes for 15375 observations

df1 = pd.DataFrame (signal_raw_1)  # pandas df for 1st electrode
df1.shape
df1.iloc [:, :5].describe ()
df1.isnull ().sum ().sum ()
df2 = pd.DataFrame (signal_raw_2)  # pandas df for 2nd electrode

df_cmb = (df1 + df2) / 2

# %% Train/Test split

X_train_raw, X_test_raw, y_train, y_test = train_test_split (signal_raw, label_raw,
                                                     test_size = 0.33, random_state = 0)

# %% reshape input data to shape that is accepted by NN

X_train_raw = X_train_raw.reshape (X_train_raw.shape [0], 3000, 2)
X_test_raw = X_test_raw.reshape (X_test_raw.shape [0], 3000, 2)

## split the signals into 1st and 2nd signal
# training data
X_train_1_raw = X_train_raw [:, :, 0]
X_train_2_raw = X_train_raw [:, :, 1]

## test data
X_test_1_raw = X_test_raw [:, :, 0]
X_test_2_raw = X_test_raw [:, :, 1]

# %% Standardise the data

# signal 1
scaler_sig1 = StandardScaler ()

X_train_1_raw = scaler_sig1.fit_transform (X_train_1_raw)
X_test_1_raw = scaler_sig1.transform (X_test_1_raw)

# signal 2
scaler_sig2 = StandardScaler ()

X_train_2_raw= scaler_sig2.fit_transform (X_train_2_raw)
X_test_2_raw = scaler_sig2.transform (X_test_2_raw)

# merge different signals together
X_train_raw = np.stack ((X_train_1_raw, X_train_2_raw), axis = 2)
X_test_raw = np.stack ((X_test_1_raw, X_test_2_raw), axis = 2)

#%% SPECTOGRAMS

print(signal_spect.shape)

X_train_spec, X_test_spec, y_train, y_test = train_test_split (signal_spect, label_raw,
                                                     test_size = 0.33, random_state = 0)

#%% reshape input data to shape that is accepted by NN

X_train_spec = X_train_spec.reshape (X_train_spec.shape [0], 100, 30, 2)
X_test_spec = X_test_spec.reshape (X_test_spec.shape [0], 100, 30, 2)

## split the signals into 1st and 2nd signal
# training data
X_train_1_spec = X_train_spec [:, :, :, 0]
X_train_2_spec = X_train_spec [:, :, :, 1]

## test data
X_test_1_spec = X_test_spec [:, :, :, 0]
X_test_2_spec = X_test_spec [:, :, :, 1]


# %% Standardise the data



# %% hot label the labels
y_train_hot = to_categorical (y_train, num_classes = 6)
y_test_hot = to_categorical (y_test, num_classes = 6)