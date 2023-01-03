import os
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
yf.pdr_override()
from datetime import date
from pandas_datareader import data as pdr
import os
import tensorflow as tf

def predict(data):
    prediction = model(data) 
    return prediction


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
directory = 'models'
top5 = {}
for file in os.listdir(directory):
    ticker = file.split(".")[0]
    try:
        model = load_model(f'models/{ticker}.h5', compile=False)
        today = date.today()
        data = pdr.get_data_yahoo(ticker, start="1970-01-01" , end=today)
        data = data.iloc[:,0:5].astype('float32').values
        max_val = data.max()
        min_val = data.min()
        data=(data-min_val)/(max_val-min_val)
        data = data[-60:,:]
    except:
        print(f"Could not find the ticker {ticker}.")
        continue
    x = []
    x.append(data)
    dataPredict = np.array(x)



    prediction = predict(dataPredict) 

    
    
    prediction = prediction * (max_val - min_val) + min_val

    graph = prediction
    prediction = prediction[0][0]
    data = data * (max_val - min_val) + min_val
    current = data[-1][2]
    percent_increase = (prediction - current) / current * 100
    if prediction > current:
        # prediction = model(data, verbose = 0) 
        top5[ticker] = {'percent_increase': percent_increase}

sorted_stocks = sorted(top5.items(), key=lambda x: x[1]['percent_increase'], reverse=True)

for stock in sorted_stocks[:5]:
    name = stock[0]
    percent_increase = stock[1]['percent_increase']
    print(f'{name}: percent_increase={percent_increase}')

    
    