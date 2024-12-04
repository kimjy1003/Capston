import pyupbit
import pandas as pd

# 데이터 로드
ticker = 'KRW-BTC'
interval = 'minute15'
to = '2024-11-01 00:00'
count = 2991
data = pyupbit.get_ohlcv(ticker=ticker, interval=interval, to=to, count=count)
data['OT'] = (data['high'] + data['low']) / 2

# day / minute1 / minute3 / minute5 / minute10 / minute15 / minute30 / minute60 / minute240 / week / month


print(data)


data.to_csv("C:\\Users\\danyj\\Desktop\\VSCode\\RevIN\\baselines\\SCINet\\data\\test\\test.csv", index = True)