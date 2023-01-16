import pandas as pd

ticker = "TSLA"
file = f'intraday/data/{ticker}.csv'
df = pd.read_csv(file)
df = df.iloc[:,[0,4,5]]

close = df.iloc[:,1].values.tolist()
calls = []
for i in range(0, len(close) - 1):
    if (close[i + 1] > close[i]*1.05):
        calls.append(1)
    elif (close[i + 1] < close[i]*0.95):
        calls.append(-1)
    else:
        calls.append(0)
calls.append(0)
df.insert(len(df.columns), "<Calls>", calls)
df.to_csv(f'intraday/data/{ticker}_TEST.csv', index=False)
