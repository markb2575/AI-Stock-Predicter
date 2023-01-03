from datetime import date
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()
import bs4 as bs
import requests

today = date.today()

def getData(ticker):
    data = pdr.get_data_yahoo(ticker, start="1970-01-01" , end=today)
    data.to_csv("data/" + ticker + ".csv")



resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
soup = bs.BeautifulSoup(resp.text, 'lxml')
table = soup.find('table', {'class': 'wikitable sortable'})


tickers = []

for row in table.findAll('tr')[1:]:
    ticker = row.findAll('td')[0].text
    tickers.append(ticker)

tickers = [s.replace('\n', '') for s in tickers]

print(tickers)

for ticker in tickers:
    getData(ticker)
