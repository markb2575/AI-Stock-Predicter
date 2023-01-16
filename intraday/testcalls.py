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


ticker = "TSLA_TEST"
print(ticker)
# if (ticker <= "AAPL"):
#     continue
file = f'intraday/data/{ticker}.csv'
df = pd.read_csv(file)
df = df.iloc[:,1:]
print(df.head())
data = df.astype('float32').values
# print(df.head())

# open, high, low, close, volume
scaled = data

# scale the close from 0-1
print("close:")
print(data[:,0].max())
print(data[:,0].min())
max_close = data[:,0].max()
min_close = data[:,0].min()
scaled[:,0]=(data[:,0]-min_close)/(max_close-min_close)

# scale the volume from 0-1
print("volume:")
print(data[:,1].max())
print(data[:,1].min())
max_volume = data[:,1].max()
min_volume = data[:,1].min()
scaled[:,1]=(data[:,1]-min_volume)/(max_volume-min_volume)

# split the data into training and testing data
index = int(scaled.shape[0] * 0.7)
datatrain = scaled[:index,:]
datatest = scaled[index:,:]

# function to turn dataset into X and Y
def create_dataset(dataset):
    dataX, dataY = [], []
    for i in range(48, len(dataset)-12):
        a = dataset[i-48:i,[0,1]]
        # print(a)
        dataX.append(a)
        a = dataset[i,2]
        # print(a)
        dataY.append(a)
    return np.array(dataX), np.array(dataY)
    
# turn the training and testing datasets into X and Y
trainX, trainY = create_dataset(datatrain)
testX, testY = create_dataset(datatest)

# neural network structure
model = Sequential()
model.add(LSTM(units=50, input_shape=(48, 2), return_sequences=True))
model.add(Dropout(0.2))
# model.add(LSTM(units=3, return_sequences=True))
# model.add(Dropout(0.2))
model.add(LSTM(units=30, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))
# model.summary()
# compile and train the model
model.compile(optimizer='adam', loss='mse')
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min', verbose=1)
hist = model.fit(trainX, trainY, batch_size=64, epochs=1000, verbose=2, validation_data=[testX,testY],callbacks=[callback])
model.save(f'intraday/models/{ticker}.h5')

print(model.predict(testX))
print(testY)