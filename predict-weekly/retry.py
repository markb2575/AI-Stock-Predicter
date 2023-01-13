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

# select a ticker to predict
ticker = "AMD"

# download the ticker data
today = date.today()
df = pdr.get_data_yahoo(ticker, start="1970-01-01" , end=today)
df.columns = df.columns.str.replace(' ', '')

# select only useful data from dataframe and convert to numpy array
df = df.iloc[:,3]
data = df.astype('float32').values

# scale the data from 0-1
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data.reshape(-1,1))

# split the data into training and testing data
index = int(scaled.shape[0] * 0.8)
datatrain = scaled[:index,:]
datatest = scaled[index:,:]

# function to turn dataset into X and Y
def create_dataset(dataset):
    dataX, dataY = [], []
    for i in range(60, len(dataset)):
        a = dataset[i-60:i,:]
        dataX.append(a)
        a = dataset[i,0]
        dataY.append(a)
    return np.array(dataX), np.array(dataY)
    
# turn the training and testing datasets into X and Y
trainX, trainY = create_dataset(scaled)
# testX, testY = create_dataset(datatest)

# neural network structure
model = Sequential()
# model.add(LSTM(units=100, input_shape=(trainX.shape[1], 1), return_sequences=True))
# model.add(Dense(trainX.shape[1]))
# model.add(Dropout(0.2))
# model.add(LSTM(units=100,return_sequences=False))
# model.add(Dropout(0.2))
# model.add(Dense(units=1))
model.add(LSTM(250, return_sequences=True, input_shape=(trainX.shape[1], 1)))
model.add(LSTM(250, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
# model.summary()

# compile and train the model
model.compile(optimizer='adam', loss='mse')
# callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min', verbose=1)
hist = model.fit(trainX, trainY, batch_size=1, epochs=3, verbose=2)

model.save('models_new.h5')

# predict on the test set and display the error
# predictions = model.predict(testX)
# predictions = scaler.inverse_transform(predictions)
# rmse = np.sqrt(np.mean(predictions - testY)**2)
# print(rmse)
# i=0
# for prediction in predictions:
#     i+=1
# print(i)
# i = 0
# for x in testY:
#     i+=1
# print(i)








