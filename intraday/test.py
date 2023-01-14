import requests
import csv
import pandas as pd
import random
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

file = 'intraday/data/NEW_SPY.csv'
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
df = df.iloc[:,1:6]
# print(df.head())
data = df.astype('float32').values
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

data = scaled
data = data[df.shape[0]-48:df.shape[0]]
# print(data)
x = []
x.append(data)
dataPredict = np.array(x)
prediction = model.predict(dataPredict)
prediction = prediction * (max_close - min_close) + min_close
data = data * (max_close - min_close) + min_close
diff = ((prediction[0][0] - data[data.shape[0]-1][3])/data[data.shape[0]-1][3]) * 100
print(f'{diff:.2}% difference.')
plt.plot(data[:,3], color = 'black', label = 'Stock Price')
plt.plot(data.shape[0],data[data.shape[0]-1][3], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red") #current price
plt.plot(data.shape[0]+24,prediction[0][0], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red")      #predicted price
plt.show()



# plt.plot(data[:,3], color = 'black', label = 'Stock Price')
# plt.plot(rand,data[-1][3], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red")
# plt.plot(rand+24,prediction[0][0], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red")
# plt.show()
