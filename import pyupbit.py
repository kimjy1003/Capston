import pyupbit

# 데이터 로드
ticker = 'KRW-BTC'
interval = 'minute1'
to = '2024-09-30 09:00'
count = 2557
data = pyupbit.get_ohlcv(ticker=ticker, interval=interval, to=to, count=count)
data['middle'] = (data['high'] + data['low']) / 2