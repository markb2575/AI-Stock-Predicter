import requests
import pandas
import csv

# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
year = 1
month = 10
with open('intraday/data/test.csv', 'a') as f:
    for year in range (year,3):
        
        for month in range(month,13):

            url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol=SPY&interval=5min&slice=year{year}month{month}&apikey=Q7ORN7T6T1JMH3PB'
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