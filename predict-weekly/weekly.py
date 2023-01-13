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
def normalize(data):
    max_val = data.max()
    min_val = data.min()
    data=(data-min_val)/(max_val-min_val)
    return data

def create_dataset(dataset):
    dataX, dataY = [], []
    # for i in range(200, len(dataset)):
    for i in range(60, len(dataset) - 30):
        # print(i)
        # print(dataset[i-60:i,:])
        # print(dataset[i:i+30,1])
        a = dataset[i-60:i,:]
        dataX.append(a)
        a = dataset[i:i+30,1]
        dataY.append(a)
    return np.array(dataX), np.array(dataY)

# def create_dataset_with_test(dataset):
#     dataX, dataY = [], []
#     # for i in range(200, len(dataset)):
#     for i in range(60, len(dataset) - 28):
#         a = dataset[i-60:i,:]
#         dataX.append(a)
#         dataY.append([dataset[i + 7,0],dataset[i + 14,0],dataset[i + 21,0],dataset[i + 28,0]])
#     return np.array(dataX[0:int(len(dataX) * .75)]), np.array(dataY[0:int(len(dataY) * .75)]),np.array(dataX[int(len(dataX) * .75):1]),np.array(dataY[int(len(dataY) * .75):1])



os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
while (1):
    # ticker = input("Enter a ticker to predict or type 'quit' to quit: ")
    ticker = "AMD"
    if ticker == "quit":
        break
    ticker = ticker.upper()
    today = date.today()
    df = pdr.get_data_yahoo(ticker, start="1970-01-01" , end=today)

    df.columns = df.columns.str.replace(' ', '')
    data = df.iloc[:,0:5].astype('float32').values


    max_val = data.max()
    min_val = data.min()
    data=(data-min_val)/(max_val-min_val)

    # datatest = data[2000:]
    # data = data[:2000]
    # print(data)
    # print("======")
    # print(datatest)
    # index = int(data.shape[0] * 0.75)
    # print(index)
    # print(int(len*0.75))
    # datatrain = data[:index,:]
    # datatest = data[index:,:]
    # print(datatest)
    # print("========")
    # print(datatrain)
    trainX, trainY = create_dataset(data)
    # testX, testY = create_dataset(datatest)
    # print("train")
    # print(trainX)
    # print("test")
    # print(testX)
    model = Sequential()
    # model.add(LSTM(1000, return_sequences=True, input_shape=(60,5))) 
    # model.add(Dense(500))
    # model.add(Dropout(0.2))
    # model.add(LSTM(1000, return_sequences=False))
    # model.add(Dropout(0.5))
    # model.add(Dense(30))
    model.add(LSTM(100, return_sequences=True, input_shape=(60,5)))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=30))
    model.compile(loss='mse', optimizer='adam')
    t0 = time.time()
    # callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, mode='min', verbose=1)
    hist = model.fit(trainX, trainY, epochs=20, batch_size=64, validation_data=[testX,testY], verbose=2)
    model.save(f'models_new/{ticker}.h5')
    print("Training time:", time.time()-t0)

    plt.plot(hist.history['loss'], label='loss')
    # plt.plot(hist.history['val_loss'], label='val_loss')
    plt.legend()
    plt.show()
    today = date.today()
    data = pdr.get_data_yahoo(ticker, start="1970-01-01" , end=today)
    data = data.iloc[:,0:5].astype('float32').values
    rand = random.randint(1000,(data.shape[0]-1000)) * -1
    print(rand)
    temp = data[rand:rand+30:,1]
    max_val = data.max()
    min_val = data.min()
    data=(data-min_val)/(max_val-min_val)

    data = data[rand-60:rand,:]
    x = []
    x.append(data)
    dataPredict = np.array(x)

    print("Predicting...")

    prediction = model.predict(dataPredict) 
    prediction = prediction * (max_val - min_val) + min_val
    data = data * (max_val - min_val) + min_val

    prediction = prediction[0]

    totaldiff = 0
    for i in range(30): 
        diff = prediction[i] - temp[i]
        print(f'{diff:.2f}% difference')
        totaldiff += diff
    totaldiff /= 30
    print(f'{totaldiff:.2f}% average difference')

    x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
    plt.plot(x,temp, color = 'black', label = 'Stock Price')
    plt.plot(x,prediction, color = 'blue', label = 'Prediction Price')

    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
    break

