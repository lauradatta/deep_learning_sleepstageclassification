# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
#%%
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import acf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras import layers

#%%

#set working directory
os.chdir("D:/Laura/Tilburg/Deep Learning II/Assignment/Deep_learning")

# Load data
with open('Data_Raw_signals.pkl', 'rb') as f:
    raw = pickle.load(f)

with open('Data_Spectrograms.pkl', 'rb') as f:
    spect = pickle.load(f)

    
#%%   
# Put data in good format

dt_raw = np.array(raw[0]) # full data
lbl_raw = np.array(raw[1]) # labels

dt_spect = np.array(spect[0]) # full data
lbl_spect = np.array(spect[1]) # labels


## Look at Raw signals
elct1 = dt_raw[:,0] # first electrode
elct2 = dt_raw[:,1] # second electrode

# elct1[:,0] elct2[:,0] - first timestep; elct1[0,:] elct2[0,:] - first person

#%%
# Visualize the data
p1e1 = elct1[0,:]
plt.plot(p1e1) # electrode 1 for first person
plt.plot(elct2[0,:]) # electrode 2 for first person

#%%
# checking autocorrelation
acf(p1e1)
acf(elct2[0,:])
autocorrelation_plot(p1e1)
autocorrelation_plot(elct2[0,:])

#%%
##Look at Spectograms

spec1 = dt_spect[:,0] #first electrode
spec2 = dt_spect[:,1] #second electrode

#%%
#visualise spectograms for first person

pers1_spec1 = spec1[0,:]
plt.plot(pers1_spec1)

pers1_spec2 = spec2[0,:]
plt.plot(pers1_spec2)


#%%
# checking basics

df1 = pd.DataFrame(elct1) # row=person, column=timestep
df1.iloc[:,:5].describe()
n df1.isnull().sum().sum()
df2 = pd.DataFrame(elct2)
df_cmb = (df1 + df2)/2

#%%
### Train/Test split ###

X_train, X_test, y_train, y_test = train_test_split(df_cmb, lbl_raw,
                                                    test_size=0.33, random_state=0)


#%%
model = Sequential()

model.add(layers.Conv1D(8, (8), activation='relu', input_shape=(10301,3000)))
model.add(layers.MaxPooling1D(8))
model.add(layers.Conv1D(16, (8), activation='relu'))
model.add(layers.MaxPooling1D(8))
model.add(layers.Conv1D(32, (8), activation='relu'))
model.add(layers.MaxPooling1D(8))


model.summary()


