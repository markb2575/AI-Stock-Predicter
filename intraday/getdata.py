import requests
import pandas as pd
import csv
import time
import os
# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
tickers = ['AAPL', 'AMD', 'AMZN', 'META', 'NVDA', 'TSLA']
year = 1
month = 1
cooldown = 5
for ticker in tickers:
    with open(f'intraday/data/{ticker}_initial.csv', 'a') as f:
        for year in range (year,2):
            
            for month in range(month,13):
                if cooldown == 0:
                    time.sleep(70)
                    cooldown = 5
                cooldown-=1
                url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol={ticker}&interval=5min&adjusted=false&slice=year{year}month{month}&apikey=Q7ORN7T6T1JMH3PB'
                print(f'{year} {month}')
                with requests.Session() as s:
                    download = s.get(url)
                    decoded_content = download.content.decode('utf-8')
                    cr = csv.reader(decoded_content.splitlines(), delimiter=',')
                    my_list = list(cr)
                    
                    for row in my_list:
                        f.write(row.__str__() + '\n')
                month+=1
            year+=1
            month = 1
        f.close()
    pd.read_csv(f'intraday/data/{ticker}_initial.csv').iloc[::-1].to_csv(f'intraday/data/{ticker}.csv', index=False)
    os.remove(f'intraday/data/{ticker}_initial.csv')



