##% run preprocessing module
from preprocessing import X_train_spec, X_test_spec, y_train_hot, y_test_hot

#import packages
# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from keras import Model, layers
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Flatten, TimeDistributed, Input, Dropout, Reshape, Input, Dropout, Conv1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, concatenate, SpatialDropout1D, Bidirectional
from keras.optimizers import Adam
import logging

# Remove some unwanted warnings
logging.getLogger('tensorflow').disabled = True

