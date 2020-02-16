import os
import pickle
import numpy as np
from keras import models
from keras import layers
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# set working directory
os.chdir (r"C:\Users\mauri\OneDrive\Documenten\Master Data Science\VAKKEN\Deep Learning\DL_data")

# Load data
with open('Data_Raw_signals.pkl', 'rb') as f:
    raw = pickle.load(f)

with open('Data_Spectrograms.pkl', 'rb') as f:
    spect = pickle.load(f)

# split the raw data into signal and labels
signal = np.array(raw[0])
labels = np.array(raw[1])
labels = labels[:]
labels.shape

#for now only work with 1 signal:
signal1 = signal[:,0,:]

#Split data into train and testset
X_train, X_test, y_train, y_test = train_test_split (signal1, labels,
                                                     test_size = 0.33, random_state = 0)

#reshape input data to shape that is accepted by NN
X_train1 = X_train.reshape(10301, 3000,1)
X_test1 = X_test.reshape(X_test.shape[0], 3000,1)

#hotlabel
y_train_hot = to_categorical(y_train, num_classes=6)
y_test_hot = to_categorical(y_test, num_classes=6)


def base_model():
    model = models.Sequential()
    model.add (layers.Conv1D (8, 8, activation = 'relu', input_shape = (3000, 1)))
    model.add (layers.MaxPooling1D (8))
    model.add (layers.Conv1D (16, 8, activation = 'relu'))
    model.add (layers.MaxPooling1D (8))
    model.add (layers.Conv1D (32, (8), activation = 'relu'))
    model.add (layers.MaxPooling1D (8))
    model.add (layers.Flatten ())
    model.add (layers.Dense (64, activation = 'relu'))
    model.add (layers.Dense (6, activation = 'softmax'))
    model.compile (loss = 'categorical_crossentropy',
                   optimizer = 'rmsprop',
                   metrics = ['accuracy'])
    return model

model = base_model()
model.summary()

model.fit(X_train1,y_train_hot,
          epochs=10,
          batch_size=100,
          verbose = 1)

cat, test_acc = model.evaluate(X_test1, y_test_hot, batch_size=100)
print(test_acc)