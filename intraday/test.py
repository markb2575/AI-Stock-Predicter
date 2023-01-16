import requests
import csv
import pandas as pd
import random
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import yfinance

ticker = "AAPL"
file = f'intraday/data/{ticker}.csv'
df = pd.read_csv(file)
df = df.iloc[:,1:6]
# print(df.head())
data = df.astype('float32').values
# print(df.head())

# open, high, low, close, volume
scaled = data

# scale the open from 0-1
max_open = data[:,0].max()
min_open = data[:,0].min()
scaled[:,0]=(data[:,0]-min_open)/(max_open-min_open)

# scale the high from 0-1
max_high = data[:,1].max()
min_high = data[:,1].min()
scaled[:,1]=(data[:,1]-min_high)/(max_high-min_high)

# scale the low from 0-1
max_low = data[:,2].max()
min_low = data[:,2].min()
scaled[:,2]=(data[:,2]-min_low)/(max_low-min_low)

# scale the close from 0-1
max_close = data[:,3].max()
min_close = data[:,3].min()
scaled[:,3]=(data[:,3]-min_close)/(max_close-min_close)

# scale the volume from 0-1
max_volume = data[:,4].max()
min_volume = data[:,4].min()
scaled[:,4]=(data[:,4]-min_volume)/(max_volume-min_volume)


model = load_model(f'intraday/models/{ticker}.h5', compile=False)
df = yfinance.download(ticker, period="60d", interval="5m")
df.columns = df.iloc[0]
df = df[1:]
df = df.iloc[::-1]
df = df.iloc[:,1:6]
print(df.head())
data = df.astype('float32').values
print(data.shape)

scaled = data

# scale the open from 0-1
scaled[:,0]=(data[:,0]-min_open)/(max_open-min_open)
# scale the high from 0-1
scaled[:,1]=(data[:,1]-min_high)/(max_high-min_high)
# scale the low from 0-1
scaled[:,2]=(data[:,2]-min_low)/(max_low-min_low)
# scale the close from 0-1
scaled[:,3]=(data[:,3]-min_close)/(max_close-min_close)
# scale the volume from 0-1
scaled[:,4]=(data[:,4]-min_volume)/(max_volume-min_volume)

bal = 500.0
for i in range(100):
    data = scaled
    temp = data * (max_close - min_close) + min_close
    rand = random.randint(50,(data.shape[0]-50))
    data = data[rand-48:rand]
    # print(data)
    x = []
    x.append(data)
    dataPredict = np.array(x)
    prediction = model.predict(dataPredict, verbose=0)
    prediction = prediction * (max_close - min_close) + min_close
    data = data * (max_close - min_close) + min_close
    diff = ((prediction[0][0] - temp[rand,3])/temp[rand,3]) * 100
    # print(diff)
    if (diff > 0):
        bal-=temp[rand,3]
        if (bal < 0):
            print("you have no more money to trade")
            break

        bal+=temp[rand+12,3]
    
    # print(f'{diff:.2}% difference.')
    # plt.plot(temp[:,3], color = 'black', label = 'Stock Price')
    # plt.plot(rand,temp[rand,3], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red") #current price
    # plt.plot(rand+12,prediction[0][0], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red") #predicted price
    # plt.xlim(rand-50,rand+50)
    # plt.ylim(min(prediction[0][0],data[-1,3])-5,max(prediction[0][0],data[-1,3])+5)
    # plt.show()
print(bal)