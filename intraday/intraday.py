from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import date
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()
import bs4 as bs
import requests
import random
from sklearn.preprocessing import MinMaxScaler

file = 'intraday/data/NEW_SPY.csv'
df = pd.read_csv(file)
df = df.iloc[:,1:5]
data = df.astype('float32').values
print(df.head())

# scale the data from 0-1
max_val = data.max()
min_val = data.min()
scaled=(data-min_val)/(max_val-min_val)

# split the data into training and testing data
index = int(scaled.shape[0] * 0.6)
datatrain = scaled[:index,:]
datatest = scaled[index:,:]

# function to turn dataset into X and Y
def create_dataset(dataset):
    dataX, dataY = [], []
    for i in range(48, len(dataset)-24):
        a = dataset[i-48:i]
        dataX.append(a)
        a = dataset[i+24,3]
        dataY.append(a)
    return np.array(dataX), np.array(dataY)
    
# turn the training and testing datasets into X and Y
trainX, trainY = create_dataset(datatrain)
testX, testY = create_dataset(datatest)

# neural network structure
model = Sequential()
model.add(LSTM(units=10, input_shape=(48, 4), return_sequences=True))
model.add(LSTM(units=10, return_sequences=False))
model.add(Dense(units=1))

# compile and train the model
model.compile(optimizer='adam', loss='mse')
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min', verbose=1)
hist = model.fit(trainX, trainY, batch_size=8, epochs=1000, verbose=2, validation_data=[testX,testY],callbacks=[callback])
model.save('intraday/SPY.h5')

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend(['Loss'],['Val_Loss'])
plt.show()