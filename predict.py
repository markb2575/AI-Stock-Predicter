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

    print("Predicting...")
    prediction = model.predict(dataPredict) 
    prediction = prediction * (max_val - min_val) + min_val
    print(prediction)
    graph = prediction
    prediction = prediction[0][0]
    data = data * (max_val - min_val) + min_val
    current = data[-1][2]
    print(f"Prediction: {prediction:.2f} Current: {current:.2f}")
    if prediction > current:
        print(f"{(prediction - current) / current * 100:.2f}% increase in the next 30 days.")
    else:
        print(f"{(prediction - current) / current * -100:.2f}% decrease in the next 30 days.")

    # plt.plot(line, data[line][2], marker="o", markersize=5, markeredgecolor="blue", markerfacecolor="blue")
    plt.plot(data[-60:,2], color = 'black', label = 'Actual Stock')
    plt.plot(data.shape[0]+30, graph, marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red")
    # plt.xlim(line-180,line+180)
    plt.title(f"{ticker} Prediction")
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()


# print(prediction)


