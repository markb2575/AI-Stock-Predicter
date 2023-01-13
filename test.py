from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
yf.pdr_override()
from datetime import date
from pandas_datareader import data as pdr
import os
import random

model = load_model('models_new.h5', compile=False)
today = date.today()
data = pdr.get_data_yahoo("AMD", start="1970-01-01" , end=today)
data = data.iloc[:,3].astype('float32').values
rand = random.randint(1000,(data.shape[0]-1000)) * -1
temp = data[rand:rand+30]
max_val = data.max()
min_val = data.min()
data=(data-min_val)/(max_val-min_val)

data = data[rand-60:rand]
x = []
x.append(data)
dataPredict = np.array(x)

for i in range(60):
    prediction = model.predict(dataPredict)
    if i == 0 or i == 59:
         print(f'the prediction was {prediction}')
    dataPredict[0] = np.roll(dataPredict[0],-1)
    dataPredict[0][-1] = prediction


dataPredict = dataPredict * (max_val - min_val) + min_val
data = data * (max_val - min_val) + min_val

totaldiff = 0
for i in range(30): 
    diff = (dataPredict[0][i] - temp[i])/temp[i]
    diff*=100
    print(f'{diff:.2f}% difference')
    totaldiff += diff
# totaldiff /= 30
# print(f'{totaldiff:.2f}% average difference')

x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
plt.plot(x,temp, color = 'black', label = 'Stock Price')
plt.plot(x,dataPredict[0][:30], color = 'blue', label = 'Prediction Price')
plt.show()

