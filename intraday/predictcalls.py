import requests
import csv
import pandas as pd
import random
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import yfinance





ticker = "TSLA"
model = load_model("intraday/models/TSLA_TEST.h5")
df = yfinance.download(ticker, period="60d", interval="5m")
# df.columns = df.iloc[0]
# df = df[1:]
df = df.iloc[:,[3,5]]
print(df)
data = df.astype('float32').values
print(data.shape)
temp = data
print("close:")
print(data[:,0].max())
print(data[:,0].min())
max_close = data[:,0].max()
min_close = data[:,0].min()


# scale the volume from 0-1
print("volume:")
print(data[:,1].max())
print(data[:,1].min())
max_volume = data[:,1].max()
min_volume = data[:,1].min()

data[:,0] = (data[:,0]-min_close)/(max_close-min_close)
data[:,1] = (data[:,1]-min_volume)/(max_volume-min_volume)
bal = 1000
for i in range(48, len(data)-12):
# print(f'{diff:.2}% difference.')
    if i == len(data)-13:
        stocks = 0
        bal+=(stocks*temp[i,0])
    if i % 100 == 0:
        print(i)
    b = []
    a = data[i-48:i]
    # print(a)
    b.append(a)
    dataPredict = np.array(b)
    prediction = model.predict(dataPredict,verbose=0)
    stocks = 0
    print(prediction)
    if prediction > 0.5:
        prediction = 1
        print("wants to buy stock")
        if (bal > temp[i,0]):
            print("buying stock")
            stocks+=1
            bal-=temp[i,0]
    elif prediction < -0.5:
        prediction = -1
        print("wants to sell stock")
        if stocks != 0:
            print("selling stock")
            stocks-=1
            bal+=temp[i,0]
    else:
        prediction = 0
    
    # plt.plot(temp[:,1], color = 'black', label = 'Stock Price')
    # plt.plot(i,temp[i,0], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red") #current price
    # plt.plot(i+12,prediction, marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red") #predicted price
    # plt.xlim(i-50,i+50)
    # plt.ylim(min(prediction,data[-1,1])-5,max(prediction,data[-1,1])+5)
    # plt.show()
print(bal)