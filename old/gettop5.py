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
directory = 'models_new'
week1 = {}
week2 = {}
week3 = {}
week4 = {}

total = len([entry for entry in os.listdir(directory) if os.path.isfile(os.path.join(directory, entry))])
num = 0
for file in os.listdir(directory):
    print(f'{num}/{total}')
    ticker = file.split(".")[0]
    try:
        model = load_model(f'models_new/{ticker}.h5', compile=False)
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


    prediction1 = prediction[0][0]
    prediction2 = prediction[0][1]
    prediction3 = prediction[0][2]
    prediction4 = prediction[0][3]
    data = data * (max_val - min_val) + min_val
    current = data[-1][2]
    percent_increase1 = (prediction1 - current) / current * 100
    percent_increase2 = (prediction2 - current) / current * 100
    percent_increase3 = (prediction3 - current) / current * 100
    percent_increase4 = (prediction4 - current) / current * 100
    if prediction1 > current:
        # prediction = model(data, verbose = 0) 
        week1[ticker] = {'percent_increase': percent_increase1}
    if prediction2 > current:
        # prediction = model(data, verbose = 0) 
        week2[ticker] = {'percent_increase': percent_increase2}
    if prediction3 > current:
        # prediction = model(data, verbose = 0) 
        week3[ticker] = {'percent_increase': percent_increase3}
    if prediction4 > current:
        # prediction = model(data, verbose = 0) 
        week4[ticker] = {'percent_increase': percent_increase4}    
    num+=1

sorted_stocks1 = sorted(week1.items(), key=lambda x: x[1]['percent_increase'], reverse=True)
sorted_stocks2 = sorted(week2.items(), key=lambda x: x[1]['percent_increase'], reverse=True)
sorted_stocks3 = sorted(week3.items(), key=lambda x: x[1]['percent_increase'], reverse=True)
sorted_stocks4 = sorted(week4.items(), key=lambda x: x[1]['percent_increase'], reverse=True)

print("Week 1:")
for stock in sorted_stocks1[:5]:
    name = stock[0]
    percent_increase = stock[1]['percent_increase']
    print(f'{name}: percent_increase={percent_increase}')
print("Week 2:")
for stock in sorted_stocks2[:5]:
    name = stock[0]
    percent_increase = stock[1]['percent_increase']
    print(f'{name}: percent_increase={percent_increase}')
print("Week 3:")
for stock in sorted_stocks3[:5]:
    name = stock[0]
    percent_increase = stock[1]['percent_increase']
    print(f'{name}: percent_increase={percent_increase}')
print("Week 4:")
for stock in sorted_stocks4[:5]:
    name = stock[0]
    percent_increase = stock[1]['percent_increase']
    print(f'{name}: percent_increase={percent_increase}')

    
    