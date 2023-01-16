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

for file in os.listdir("intraday/data"):
    file = os.path.join("intraday/data", file)
    ticker = file.split("\\")[1].split(".")[0]
    print(ticker)
    # if (ticker <= "AAPL"):
    #     continue
    file = f'intraday/data/{ticker}.csv'
    df = pd.read_csv(file)
    df = df.iloc[:,1:6]
    print(df.head())
    data = df.astype('float32').values
    # print(df.head())

    # open, high, low, close, volume
    scaled = data

    # scale the open from 0-1
    print("open:")
    print(data[:,0].max())
    print(data[:,0].min())
    max_open = data[:,0].max()
    min_open = data[:,0].min()
    scaled[:,0]=(data[:,0]-min_open)/(max_open-min_open)

    # scale the high from 0-1
    print("high:")
    print(data[:,1].max())
    print(data[:,1].min())
    max_high = data[:,1].max()
    min_high = data[:,1].min()
    scaled[:,1]=(data[:,1]-min_high)/(max_high-min_high)

    # scale the low from 0-1
    print("low:")
    print(data[:,2].max())
    print(data[:,2].min())
    max_low = data[:,2].max()
    min_low = data[:,2].min()
    scaled[:,2]=(data[:,2]-min_low)/(max_low-min_low)

    # scale the close from 0-1
    print("close:")
    print(data[:,3].max())
    print(data[:,3].min())
    max_close = data[:,3].max()
    min_close = data[:,3].min()
    scaled[:,3]=(data[:,3]-min_close)/(max_close-min_close)

    # scale the volume from 0-1
    print("volume:")
    print(data[:,4].max())
    print(data[:,4].min())
    max_volume = data[:,4].max()
    min_volume = data[:,4].min()
    scaled[:,4]=(data[:,4]-min_volume)/(max_volume-min_volume)

    # split the data into training and testing data
    index = int(scaled.shape[0] * 0.7)
    datatrain = scaled[:index,:]
    datatest = scaled[index:,:]

    # function to turn dataset into X and Y
    def create_dataset(dataset):
        dataX, dataY = [], []
        for i in range(48, len(dataset)-12):
            a = dataset[i-48:i]
            dataX.append(a)
            a = dataset[i+12,3]
            dataY.append(a)
        return np.array(dataX), np.array(dataY)
        
    # turn the training and testing datasets into X and Y
    trainX, trainY = create_dataset(datatrain)
    testX, testY = create_dataset(datatest)

    # neural network structure
    model = Sequential()
    model.add(LSTM(units=50, input_shape=(48, 5), return_sequences=True))
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


