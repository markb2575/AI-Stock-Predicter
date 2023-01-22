import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
ticker = "TSLA"
file = f'intraday/data/{ticker}.csv'
df = pd.read_csv(file)
df = df.iloc[:,[0,4,5]]

close = df.iloc[:,1].values.tolist()
plt.plot(close)
plt.show()
close = savgol_filter(close, 10, 3)
plt.plot(close)
plt.show()
calls = []

for i in range(0, len(close) - 1):
    if (close[i - 1] > close[i] < close[i + 1]):
        calls.append(-1)
    elif (close[i - 1] < close[i] > close[i + 1]):
        calls.append(1)
    else:
        calls.append(0)
calls.append(0)

df.insert(len(df.columns), "<BUY>", calls)
df.to_csv(f'intraday/data/{ticker}_TEST.csv', index=False)
