import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
import matplotlib.dates as mdates
from datetime import datetime, timedelta

import plotly.plotly as py
import plotly.graph_objs as go

# Machine learning classification libraries
import importlib
from sklearn.svm import SVC, SVR
from sklearn.metrics import scorer
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor



# For data manipulation
import pandas as pd
import numpy as np

# To plot
import matplotlib.pyplot as plt
import seaborn

# To fetch data
from pandas_datareader import data as pdr
from collections import  deque
import time
from alpha_vantage.timeseries import TimeSeries
import random
import os
import sqlite3



# Get historical stock data and predict off that
# Get it into realtime
# make the model dynamic with the realtime data

# Start scanning historical news articals on companies, make summaries and key words
# look at overall market and that company's stock for changes.
# categorize those key words (good vs bad (made price go up vs down))
# word vs company stock trend
# was it already trending down/up
# word vs overall market trend
# was it already trending down/up
# synonyms
# frequency of articles on that company
# be aware of context and sayings (CEO was fired vs. CEO fired up the shareholders)
# might have to look for combinations of those key words
# look for them in other articals, make sure the categories are correct 
# start modeling based off of that 
# integrate this model into realtime model

# In terms of a human, the real time model will be the reflexes, 
# news artical model will be the brain processing info and reacting accordingly


#SVR, Regression, etc


alphakeys=['R2FFCW41HVNZ8DBN','O6EZ7OAWV5ERVK8S','DVHJN2K8OOWYEO8F','E6K5ZE32ODWBDPOT','K917CGMUX1MY3NZH','96ZN1WBIGXC6XSOT']

#Globals
indices={'DOW':'DJI', 'NASDAQ':'IXIC', 'S&P':'GSPC'}
stocks=['GOOGL','M','BAC','XOM','QQQ','V','DUK','VZ','CVX','PYPL','BRK.B','AMD','JPM']
split_percentage = 0.8

refreshData = False

graphPlots=False

def main():
    priceData={}

    if refreshData:
        priceData = getPriceData()
        for i in priceData:
            data = priceData[i]
            data.to_pickle("pickles/"+i)
    else:
        for f in os.listdir("./pickles"):
            priceData[f] = pd.read_pickle("./pickles/"+f)

    trainPriceData(priceData)

def trainPriceData(priceData):
    print("\nSVC")
    trainSVC(priceData)
    print("\nKNN")
    trainKNN(priceData)
    if graphPlots:
        input("Press any key...")



def trainKNN(priceData):
    plt.figure()
    names = []
    accuracies = []
    for ndata in priceData:
        data = priceData[ndata]
        y = np.where(data['Close'].shift(-1) > data['Close'], 1, -1)
        data['Open-Close'] = data.Open - data.Close
        data['High-Low'] = data.High - data.Low
        X = data[['Open-Close', 'High-Low']]
        split = int(split_percentage * len(data))

        # Train data set
        X_train = X[:split]
        y_train = y[:split]

        # Test data set
        X_test = X[split:]
        y_test = y[split:]

        knn=KNeighborsClassifier(n_neighbors=15, weights='uniform', algorithm='auto',leaf_size=30,\
                                 p=2,metric='minkowski',metric_params=None, n_jobs=None)
        knn.fit(X_train, y_train)

        accuracy_train = accuracy_score(y_train, knn.predict(X_train))

        accuracy_test = accuracy_score(y_test, knn.predict(X_test))


        data['Predicted_Signal'] = knn.predict(X)

        # Calculate log returns
        data['Return'] = np.log(data.Close/data.Close.shift(1))
        cum_ret = data[split:]['Return'].cumsum() * 100


        data['Strategy_Return'] = data.Return * data.Predicted_Signal.shift(1)
        cum_strat_ret= data[split:]['Strategy_Return'].cumsum() * 100
        std = cum_strat_ret.std()
        Sharpe = (cum_strat_ret - cum_ret)/std
        Sharpe=Sharpe.mean()
        #print ('Sharpe ratio: %.2f'%Sharpe)

        accuracies.append([ndata, accuracy_train * 100, accuracy_test * 100, Sharpe])
        data.Strategy_Return.iloc[split:].cumsum().plot(figsize=(10, 5))
        names.append(ndata)

    accuracies = sorted(accuracies, key=lambda so: so[2])
    for acc in accuracies:
        print("%6s   %.4f   %.4f   %.3f"%(acc[0],acc[1],acc[2],acc[3]))
    print()
    if graphPlots:
        plt.ylabel("Strategy Returns (%)")
        plt.grid(True)
        plt.show(block=False)
        plt.legend(names)
        plt.pause(0.1)
    return accuracies




def trainSVC(priceData):
    plt.figure()
    names=[]
    accuracies=[]
    for ndata in priceData:
        data=priceData[ndata]
        y = np.where(data['Close'].shift(-1) > data['Close'], 1, -1)
        data['Open-Close'] = data.Open - data.Close
        data['High-Low'] = data.High - data.Low
        X = data[['Open-Close', 'High-Low']]
        split = int(split_percentage * len(data))

        # Train data set
        X_train = X[:split]
        y_train = y[:split]

        # Test data set
        X_test = X[split:]
        y_test = y[split:]
        sv = SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True,\
                 probability=False, cache_size=200, class_weight=None, max_iter=-1,\
                 decision_function_shape='ovr', random_state=None)
        cls = sv.fit(X_train, y_train)

        accuracy_train = accuracy_score(y_train, cls.predict(X_train))

        accuracy_test = accuracy_score(y_test, cls.predict(X_test))


        data['Predicted_Signal'] = cls.predict(X)

        # Calculate log returns

        data['Return'] = np.log(data.Close / data.Close.shift(1))
        cum_ret = data[split:]['Return'].cumsum() * 100

        data['Strategy_Return'] = data.Return * data.Predicted_Signal.shift(1)
        cum_strat_ret = data[split:]['Strategy_Return'].cumsum() * 100

        data.Strategy_Return.iloc[split:].cumsum().plot(figsize=(10, 5))
        std = cum_strat_ret.std()
        Sharpe = (cum_strat_ret - cum_ret) / std
        Sharpe = Sharpe.mean()
        # print ('Sharpe ratio: %.2f'%Sharpe)

        accuracies.append([ndata, accuracy_train * 100, accuracy_test * 100, Sharpe])



        names.append(ndata)


    accuracies = sorted(accuracies,key=lambda so: so[2])
    for acc in accuracies:
        print("%6s   %.4f   %.4f   %.3f"%(acc[0],acc[1],acc[2],acc[3]))
    print()
    if graphPlots:
        plt.ylabel("Strategy Returns (%)")
        plt.grid(True)
        plt.show(block=False)
        plt.legend(names)
    return accuracies
    
    
    
    
def getPriceData():
    data={}
    for index in indices:
        while True:
            try:
                akey = random.choice(alphakeys)
                ts = TimeSeries(key=akey, output_format='pandas')
                tdata, meta_data = ts.get_intraday(symbol=indices[index], interval='5min', outputsize='full')
                break
            except ValueError:
                continue
        print(index, akey)
        tdata = tdata.rename(columns={'1. open': 'Open', '2. high': 'High', '3. low': 'Low', \
                                      '4. close': 'Close', '5. volume': 'Volume'})
        tdata = tdata.dropna()
        data[index]=tdata
    for stock in stocks:
        while True:
            try:
                akey = random.choice(alphakeys)
                ts = TimeSeries(key=akey, output_format='pandas')
                tdata, meta_data = ts.get_intraday(symbol=stock, interval='5min', outputsize='full')
                break
            except ValueError:
                continue
        print(stock, akey)
        tdata = tdata.rename(columns={'1. open': 'Open', '2. high': 'High', '3. low': 'Low', \
                                      '4. close': 'Close', '5. volume': 'Volume'})
        tdata = tdata.dropna()
        data[stock]=tdata

    return data



if __name__ == "__main__":
    main()
