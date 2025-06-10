In [48]:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import time
import datetime
import pandas as pd
import numpy as np
import pyotp
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
from sklearn.pipeline import Pipeline
In [49]:
import requests # Install requests module first.

url = "https://public.coindcx.com/market_data/candles?pair=B-BTC_USDT&interval=1m" # Replace 'SNTBTC' with the desired market pair.

response = requests.get(url)
data = response.json()
data_f=pd.DataFrame(data)
data_f
Out[49]:
	open	high	low	volume	close	time
0	109464.82	109464.83	109463.01	1.08416	109463.01	1749533160000
1	109459.92	109477.06	109459.91	5.54338	109464.82	1749533100000
2	109449.09	109460.00	109400.00	26.41650	109459.92	1749533040000
3	109500.00	109500.01	109449.09	9.06334	109449.09	1749532980000
4	109572.04	109572.05	109500.00	12.03132	109500.00	1749532920000
...	...	...	...	...	...	...
495	109192.06	109374.99	109132.00	91.67599	109281.68	1749503460000
496	108996.87	109316.33	108996.87	346.19421	109192.06	1749503400000
497	108808.33	108996.88	108808.33	129.54848	108996.87	1749503340000
498	108777.82	108808.34	108766.76	8.92314	108808.33	1749503280000
499	108771.26	108786.14	108771.25	2.93656	108777.82	1749503220000
500 rows × 6 columns
In [50]:
def ewm(data_f):
    data_f["close"] = pd.to_numeric(data_f["close"], errors='coerce')

    data_f["ewm9"] = data_f["close"].ewm(com=9).mean()
    data_f["ewm15"] = data_f["close"].ewm(com=15).mean()

    return data_f["ewm9"], data_f["ewm15"]
In [51]:
def william_p():
    close_p=data_f["close"].iloc[0]
    h_high=data_f["high"].rolling(window=14).max()
    l_low=data_f["low"].rolling(window=14).min()
    william =((h_high-close_p)/(h_high-l_low))*(-100)
    return william
In [52]:
def bollinger_band(data_f):

    data_f["close"] = pd.to_numeric(data_f["close"], errors='coerce')
    data_f["MB"] = data_f["close"].rolling(window=20).mean()                         # Middle Band (Exponential Moving Average)
    data_f["LB"] = data_f["MB"] - 2 * data_f["close"].ewm(com=9).std()         # Lower Band
    data_f["HB"] = data_f["MB"] + 2 * data_f["close"].ewm(com=9).std()         # Higher Band
    data_f["b_width"] = data_f["HB"] - data_f["LB"]                       # Band Width
    data_f.dropna(subset=["MB", "HB", "LB"], inplace=True)
    

    return data_f["LB"], data_f["HB"], data_f["MB"]
In [53]:
def volatility_percentage():
    range=data_f['high']-data_f['low']
    normal_range=range/data_f['close']
    percen_volatility=normal_range*100
    return percen_volatility
In [54]:
def ichimoku_cloud():
    conversion_line=(data_f["high"].rolling(window=9).max()+data_f["low"].rolling(window=9).min())/2
    base_line=(data_f["high"].rolling(window=26).max()+data_f["low"].rolling(window=26).min())/2
    leading_span_A=((base_line+conversion_line)/2).shift(26)
    leading_span_B=((data_f["high"].rolling(window=52).max()+data_f["low"].rolling(window=52).min())/2).shift(26)
    lagging_span=data_f["close"].shift(-26)
    return conversion_line,base_line,leading_span_A,leading_span_B,lagging_span
    
    
In [55]:
def macd(data_f):

    data_f["MACD_LINE"] = data_f["close"].ewm(span=12).mean() - data_f["close"].ewm(span=26).mean()


    data_f["SIGNAL_LINE"] = data_f["MACD_LINE"].ewm(span=9).mean()


    data_f["MACD_HIST"] = data_f["MACD_LINE"] - data_f["SIGNAL_LINE"]


    return data_f["MACD_LINE"], data_f["SIGNAL_LINE"], data_f["MACD_HIST"]
In [56]:
def parabolic_sar(data_f, af_start=0.02, af_step=0.02, af_max=0.2):
    high = data_f['high'].values
    low = data_f['low'].values
    close = data_f['close'].values
    
    length = len(data_f)
    sar = [None] * length
    trend = [None] * length

    # Initialization
    af = af_start
    ep = high[0]
    sar[1] = low[0]  # Assume starting trend is up
    trend[1] = 'up'

    for i in range(2, length):
        prev_sar = sar[i - 1]
        prev_ep = ep
        prev_af = af
        prev_trend = trend[i - 1]

        if prev_trend == 'up':
            sar_raw = prev_sar + prev_af * (prev_ep - prev_sar)
            sar[i] = min(sar_raw, low[i - 1], low[i - 2])

            if high[i] > prev_ep:
                ep = high[i]
                af = min(prev_af + af_step, af_max)
            else:
                ep = prev_ep
                af = prev_af

            if low[i] < sar[i]:
                # Trend reversal to down
                trend[i] = 'down'
                sar[i] = prev_ep
                ep = low[i]
                af = af_start
            else:
                trend[i] = 'up'

        else:  # prev_trend == 'down'
            sar_raw = prev_sar + prev_af * (prev_ep - prev_sar)
            sar[i] = max(sar_raw, high[i - 1], high[i - 2])

            if low[i] < prev_ep:
                ep = low[i]
                af = min(prev_af + af_step, af_max)
            else:
                ep = prev_ep
                af = prev_af

            if high[i] > sar[i]:
                # Trend reversal to up
                trend[i] = 'up'
                sar[i] = prev_ep
                ep = high[i]
                af = af_start
            else:
                trend[i] = 'down'

    data_f['parabolic_sar'] = sar
    data_f['trend'] = trend
    return data_f
In [57]:
def stochastic(data_f):
    # Ensure the required columns are numeric
    data_f["high"] = pd.to_numeric(data_f["high"], errors="coerce")
    data_f["low"] = pd.to_numeric(data_f["low"], errors="coerce")
    data_f["close"] = pd.to_numeric(data_f["close"], errors="coerce")


    # Calculate 14-day high and low
    data_f["high14"] = data_f["high"].rolling(window=14).max()
    data_f["low14"] = data_f["low"].rolling(window=14).min()

    denominator = data_f["high14"] - data_f["low14"]
    denominator.replace(0, pd.NA)

    # Calculate %K
    data_f["%k"] = (
        ((data_f["close"] - data_f["low14"]) / (data_f["high14"] - data_f["low14"]))
        * 100
    ).rolling(window=3).mean()

    # Handle cases where the denominator is zero
    data_f["%k"].where(data_f["high14"] != data_f["low14"], other=0)

    # Calculate %D as the 3-day rolling mean of %K
    data_f["%d"] = data_f["%k"].rolling(window=3).mean()

    # Return %K and %D
    return data_f["%k"],data_f["%d"]
In [58]:
def donchain_channel():
    highest_high=data_f["high"].rolling(window=20).max()
    lowest_low=data_f["low"].rolling(window=20).min()
    middle_band=(highest_high+lowest_low)/2
    return highest_high,lowest_low,middle_band
In [59]:
def roc(data_f):
    roc=((data_f["close"]-data_f["close"].shift(10))/data_f["close"].shift(10)*100)
    return roc
In [60]:
def momentum_indi():
    momentum=(data_f["close"]/data_f["close"].shift(10)-1)*100
    return momentum
In [61]:
def commodity_ci():
    typical_price=(data_f["close"]+data_f["low"]+data_f["high"])/3
    sm_typical_price=typical_price.rolling(window=20).mean()
    mean_deviation = typical_price.rolling(window=20).apply(lambda x: (abs(x - x.mean())).mean(), raw=False)
    commodity_channel_index=(typical_price-sm_typical_price)/(0.015*mean_devation)
    return commodity_channel_index
In [62]:
def rsi(data_f):
    data_f["change"]=data_f["close"]-data_f["close"].shift(1)
    data_f["gain"]=np.where(data_f["change"]>=0,data_f["change"],0)
    data_f["loss"]=np.where(data_f["change"]<0,-1*data_f["change"],0)
    data_f["avg_gain"]=data_f["gain"].ewm(alpha=1/14,min_periods=14).mean()
    data_f["avg_loss"]=data_f["loss"].ewm(alpha=1/14,min_periods=14).mean()
    data_f["rsi"]=100-(100/(1+data_f["avg_gain"]/data_f["avg_loss"]))
    return data_f["rsi"]
In [63]:
def atr_adx(data_f):
    # Ensure all data is numeric, coercing errors to NaN
    data_f = data_f.apply(pd.to_numeric, errors='coerce')

    data_f["h-l"] = data_f["high"] - data_f["low"]
    data_f["h-cp"] = data_f["high"] - data_f["close"].shift(1)
    data_f["l-cp"] = data_f["low"] - data_f["close"].shift(1)
    data_f["tr"] = data_f[["h-l", "h-cp", "l-cp"]].max(axis=1)
    data_f["$High"]=data["high"]-data_f["high"].shift(1)
    data_f["$Low"]=data["low"].shift(1)-data_f["low"]
    data_f["atr"] = data_f["tr"].rolling(14).mean()

    return data_f["atr"], data_f["tr"]
In [85]:
def pivot_standard():
    central_pivot=(data_f["low"]+data_f["high"]+data_f["close"])/3
    first_resistance=2*central_pivot-data_f["low"]
    first_support=2*central_pivot-data_f["high"]
    second_resistance=central_pivot+data_f["high"]-data_f["low"]
    second_support=central_pivot-data_f["high"]+data_f["low"]
    third_resistance=2*central_pivot+data_f["high"]-2*data_f["low"]
    third_support=2*central_pivot-2*data_f["high"]+data_f["low"]
    return central_pivot,first_support,second_support,third_support,first_resistance,second_resistance,third_resistance
    
In [87]:
def pivot_cammerilla():
    delta=data_f["high"]-data_f["low"]
    h1=data_f["close"]+delta*(1.1/12)
    h2=data_f["close"]+delta*(1.1/6)
    h3=data_f["close"]+delta*(1.1/4)
    h4=data_f["close"]+delta*(1.1/2)
    l1=data_f["close"]-delta*(1.1/12)
    l2=data_f["close"]-delta*(1.1/6)
    l3=data_f["close"]-delta*(1.1/4)
    l4=data_f["close"]-delta*(1.1/2)
    return l1,l2,l3,l4,h1,h2,h3,h4
In [91]:
def target_close(data_f):
    # Ensure 'intc' column is numeric
    data_f["close"] = pd.to_numeric(data_f["close"], errors='coerce')

    # Initialize 'signal' column with 0
    data_f['signal'] = 0

    # Assign signals based on correct interpretation of percentage change
    data_f.loc[data_f['close'].pct_change(2) > 0.0007, 'signal'] = 2  # Buy Put
    data_f.loc[data_f['close'].pct_change(2) < -0.0007, 'signal'] = 1  # Buy Call

    return data_f["signal"]
In [ ]:
 
In [ ]:
 
In [ ]:
 
In [ ]:
 
In [ ]:
 
In [ ]:
 
In [ ]:
 
In [ ]:
 
In [ ]:
 
In [ ]:
 
In [ ]:
 
In [ ]:
 
In [ ]:
 
In [ ]:
 
In [ ]:
 
In [ ]:
 
In [ ]:
 
In [ ]:
 
In [ ]:
 
In [ ]:
 
In [ ]:
 
In [ ]:
 
In [ ]:
 
In [ ]:
 
In [ ]:
 
In [ ]:
 
In [ ]:
 
