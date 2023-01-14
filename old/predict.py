from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
yf.pdr_override()
from datetime import date
from pandas_datareader import data as pdr
import os


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
while (1):
    ticker = input("Enter a ticker to predict or type 'quit' to quit: ")
    if ticker == "quit":
        break
    ticker = ticker.upper()
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

    print("Predicting...")
    # prediction = model.predict(dataPredict) 
    # prediction = prediction * (max_val - min_val) + min_val
    # print(prediction)
    # graph = prediction
    # prediction = prediction[0][0]
    # data = data * (max_val - min_val) + min_val
    # current = data[-1][2]
    # print(f"Prediction: {prediction:.2f} Current: {current:.2f}")
    # if prediction > current:
    #     print(f"{(prediction - current) / current * 100:.2f}% increase in the next 30 days.")
    # else:
    #     print(f"{(prediction - current) / current * -100:.2f}% decrease in the next 30 days.")

    prediction = model.predict(dataPredict) 

    
    
    prediction = prediction * (max_val - min_val) + min_val


    prediction1 = prediction[0][0]
    prediction2 = prediction[0][1]
    prediction3 = prediction[0][2]
    prediction4 = prediction[0][3]
    data = data * (max_val - min_val) + min_val
    current = data[-1][2]

    if prediction1 > current:
        # prediction = model(data, verbose = 0) 
        print(f"{(prediction1 - current) / current * 100:.2f}% increase in the next 7 days.")
    else:
        print(f"{(prediction1 - current) / current * -100:.2f}% decrease in the next 7 days.")
    if prediction2 > current:
        # prediction = model(data, verbose = 0) 
        print(f"{(prediction2 - current) / current * 100:.2f}% increase in the next 14 days.")
    else:
        print(f"{(prediction2 - current) / current * -100:.2f}% decrease in the next 14 days.")
    if prediction3 > current:
        # prediction = model(data, verbose = 0) 
        print(f"{(prediction3 - current) / current * 100:.2f}% increase in the next 21 days.")
    else:
        print(f"{(prediction3 - current) / current * -100:.2f}% decrease in the next 21 days.")
    if prediction4 > current:
        # prediction = model(data, verbose = 0) 
        print(f"{(prediction4 - current) / current * 100:.2f}% increase in the next 28 days.")
    else:
        print(f"{(prediction4 - current) / current * -100:.2f}% decrease in the next 28 days.")

    # plt.plot(line, data[line][2], marker="o", markersize=5, markeredgecolor="blue", markerfacecolor="blue")
    plt.plot(data.shape[0]+7, prediction[0][0], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red")
    plt.plot(data.shape[0]+14, prediction[0][1], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red")
    plt.plot(data.shape[0]+21, prediction[0][2], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red")
    plt.plot(data.shape[0]+28, prediction[0][3], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red")
    plt.plot(data[-60:,0], color = 'black', label = 'Stock Price')

    # plt.xlim(line-180,line+180)
    plt.title(f"{ticker} Prediction")
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()


# print(prediction)


