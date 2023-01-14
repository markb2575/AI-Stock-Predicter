import requests
import csv
import pandas as pd
import random
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

model = load_model('intraday/SPY.h5', compile=False)
url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=SPY&interval=5min&outputsize=compact&datatype=csv&apikey=Q7ORN7T6T1JMH3PB'
# url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=SPY&interval=5min&datatype=csv&apikey=Q7ORN7T6T1JMH3PB'
with requests.Session() as s:
    download = s.get(url)
decoded_content = download.content.decode('utf-8')
cr = csv.reader(decoded_content.splitlines(), delimiter=',')
df = pd.DataFrame(cr)
df.columns = df.iloc[0]
df = df[1:]
df = df.iloc[::-1]
df = df.iloc[:,1:5]
data = df.astype('float32').values

max_val = data.max()
min_val = data.min()
data=(data-min_val)/(max_val-min_val)
data = data[df.shape[0]-48:df.shape[0]]
x = []
x.append(data)
dataPredict = np.array(x)
prediction = model.predict(dataPredict)
prediction = prediction * (max_val - min_val) + min_val
data = data * (max_val - min_val) + min_val
plt.plot(data[:,3], color = 'black', label = 'Stock Price')
plt.plot(data.shape[0],data[data.shape[0]-1][3], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red")
plt.plot(data.shape[0]+24,prediction[0][0], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red")
plt.show()
# plt.plot(data[:,3], color = 'black', label = 'Stock Price')
# plt.plot(rand,data[-1][3], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red")
# plt.plot(rand+24,prediction[0][0], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red")
# plt.show()
