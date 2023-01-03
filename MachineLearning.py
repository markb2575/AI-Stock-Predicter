from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

def normalize(data):
    max_val = data.max()
    min_val = data.min()
    data=(data-min_val)/(max_val-min_val)
    return data

def create_dataset(dataset):
    dataX, dataY = [], []
    # for i in range(200, len(dataset)):
    for i in range(len(dataset)-60-1):
        a = dataset[i:(i+60)]
        dataX.append(a)
        dataY.append(dataset[i,0])

    # print("X values")
    # print(dataX)
    # print("Y values")
    # print(dataY)
    return np.array(dataX), np.array(dataY)



directory = 'data'
total = len([entry for entry in os.listdir(directory) if os.path.isfile(os.path.join(directory, entry))])
current = 0
print(f'{current}/{total}')
for file in os.listdir(directory):
    file = os.path.join(directory, file)

    print(file.split("\\")[1].split(".")[0])
    if (file.split("\\")[1].split(".")[0] < "CINF"):
        current+=1
        continue
    df = pd.read_csv(file)
    df.columns = df.columns.str.replace(' ', '')
    data = df.iloc[:,1:6].astype('float32').values
    print(file)
    max_val = data.max()
    min_val = data.min()
    data=(data-min_val)/(max_val-min_val)

    # split = int(len(data) * 0.70)
    # train = data[:split]
    # test = data[split:]
    trainX, trainY = create_dataset(data)
    # testX, testY = create_dataset(test) 
    # print("shapes")
    # print(trainX.shape)
    # print(trainX.shape[0])
    # print(trainX.shape[1])
    # trainX = np.reshape(trainX, ((200, 5)))
    # testX = np.reshape(testX, ((200, 5)))
    model = Sequential()
    model.add(LSTM(units=50, input_shape=(60, 5), return_sequences=True))
    model.add(Dense(1))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(loss='mse', optimizer='adam')
    t0 = time.time()
    hist = model.fit(trainX, trainY, epochs=100, batch_size=256, verbose=0)
    print("Training time:", time.time()-t0)
    
    model.save('models/' + file.split("\\")[1].split(".")[0] + '.h5')
    current+=1
    print(f'{current}/{total}')
    # plt.plot(hist.history['loss'])
    # plt.ylabel("Loss")
    # plt.xlabel("Epochs")
    # plt.legend(["Loss"])
    # plt.show()






# https://stackoverflow.com/questions/64639524/training-a-neural-network-with-multiple-datasets-keras
# https://github.com/lukas/ml-class/blob/master/projects/6-rnn-timeseries/train.py


