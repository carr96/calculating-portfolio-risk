import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt

tickers = ['MSFT', 'AAPL', 'CGC', 'AVGO', 'AMT']

sec_data = pd.DataFrame()

for t in tickers:
    sec_data[t] = wb.DataReader(t, data_source='iex', start='2015-1-1')['close']
    
sec_returns = np.log(sec_data / sec_data.shift(1))

sec_returns[['MSFT', 'AAPL', 'CGC', 'AVGO', 'AMT']].mean() * 250

sec_returns[['MSFT', 'AAPL','CGC', 'AVGO', 'AMT']].std() * 250 ** 0.5

weights = np.array([0.2,0.2,0.2,0.2,0.2])

pfolio_vol = (np.dot(weights.T, np.dot(sec_returns.cov() * 250, weights))) ** 0.5

print (str(round(pfolio_vol,5) * 100) + '%')