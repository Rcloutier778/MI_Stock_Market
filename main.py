from multiprocessing import Pool

from datetime import datetime, timedelta

# Machine learning classification libraries
from sklearn import preprocessing
from sklearn.svm import SVC, SVR, NuSVC, LinearSVC
from sklearn.metrics import scorer
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier

# For data manipulation
import pandas as pd
import numpy as np

# To plot
import seaborn

# To fetch data
from pandas_datareader import data as pdr
from collections import deque
import time
from alpha_vantage.timeseries import TimeSeries
import random
import os
import sqlite3
from bs4 import BeautifulSoup
import requests
import re
import urllib3.request
import Summarizer
import json
from collections import defaultdict, OrderedDict, Counter
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from newsArticles import *
import copy
import warnings
import statistics
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore",category=matplotlib.MatplotlibDeprecationWarning)
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore",category=matplotlib.MatplotlibDeprecationWarning)
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
import statistics


#warnings.filterwarnings("ignore", category=pd.DataFrame.SettingWithCopyWarning)
pd.options.mode.chained_assignment = None
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


# SVR, Regression, etc


alphakeys = ['R2FFCW41HVNZ8DBN', 'O6EZ7OAWV5ERVK8S', 'DVHJN2K8OOWYEO8F', 'E6K5ZE32ODWBDPOT',
             'K917CGMUX1MY3NZH', '96ZN1WBIGXC6XSOT', '3MJILEPY41I90ATY', 'S2NFK5KCIYCZSVO1',
             'YJKYECKVB9U1XS1Z', 'W7I10I713Y9PJURK']

##Globals
indices = {'DOW': 'DJI', 'NASDAQ': 'IXIC'}
# Relate Jpmorgan and chase bank?
stocks = {'GOOGL': 'Google', 'M': 'Macys', 'BAC': 'Bank of America', 'XOM': 'Exxon Mobil',
          'V': 'Visa', 'DUK': 'Duke Energy', 'VZ': 'Verizon', 'CVX': 'Chevron',
          'PYPL': 'Paypal', 'AMD': 'AMD', 'JPM': 'J.P. Morgan'}

# Data split percentage for train/test
split_percentage = 0.8

# Get new data
refreshData = False

# Graphs plots
graphPlots = True
matlab = False
startAfterCrash = True

def main():
    pd.options.mode.chained_assignment = None
    # From the data intervals, concat the following together to use
    if refreshData:
        print("Refreshing price data")
        # Get and save new data
        priceData = getPriceData()
        for i in priceData:
            data = priceData[i]
            data.to_pickle("pickles/" + i + ".pkl")
    
    dataconcat = ['30min', '15min']  # ,'5min','1min']#, '5min', '1min']
    # Use saved data
    print("Reading  price data")
    priceData, dailyData = readPriceData(dataconcat)
    
    # Convert data for matlab nnet
    if matlab:
        newDailyData = getArticles(priceData, dailyData, stocks)
        writeToMatlab(priceData, dailyData, newDailyData)
        return
    
    
    # TODO start after 2008 crash
    
    # Get news article data and integrate it with daily data
    newDailyData = getArticles(priceData, dailyData, stocks)

    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #    print(newDailyData['GOOGL'])
    
    dailyDataWithArticles={}
    #dailyDataNoArticles={}
    
    for stock in dailyData:
        if stock in newDailyData:
            dailyDataWithArticles[stock] = copy.deepcopy(newDailyData[stock])
           # dailyDataNoArticles[stock] = copy.deepcopy(dailyData[stock])
            

    # Intraday data prediction
    # trainSVC(priceData)
    
    # Daily data prediction
    print("Daily - SVM")
    dailyDataWithArticles = trainSVC(dailyDataWithArticles,False)
    
    print("\nDaily - NNet")
    dailyDataWithArticles = neuralNet(dailyDataWithArticles,False)

    # Daily data prediction with news articles
    print("\nDaily w/ news- SVM")
    dailyDataWithArticles = trainSVC(dailyDataWithArticles, True)

    print("\nDaily w/ news - NNet")
    dailyDataWithArticles = neuralNet(dailyDataWithArticles,True)

    if graphPlots:
        plotGraphs(dailyDataWithArticles)
    
    return 0


def plotGraphs(dailyData):
    print("Plotting graphs")
    for ndata in dailyData:
        plt.figure(ndata, figsize=(12, 7))
        data = dailyData[ndata]
        data.cum_ret.plot(color='r', label='Returns')
        data.svm_cum_strat_ret_wo.plot(color='g', label='SVM Strategy Returns w/o')#, linestyle='dashed')
        data.nnet_cum_strat_ret_wo.plot(color='y', label='NNet Strategy Returns w/o')  # , linestyle='dashed')
        #data.svm_cum_strat_ret_w.plot(color='b', label='SVM Strategy Returns w', linestyle='dashed')
        
        #data.nnet_cum_strat_ret_w.plot(color='k', label='NNet Strategy Returns w', linestyle='dashed')
        plt.title(ndata)
        plt.xlabel("Dates")
        plt.ylabel("Percentage Return")
        plt.legend(['Returns', 'SVM Strategy','NNet','SVM Strategy w','NNet w'])
        plt.grid(True)
        plt.show(block=False)

    plt.pause(0.001)
    plt.waitforbuttonpress()
    return


def trainSVC(priceData,withArticles):
    """
    Parent SVC function
    :param priceData:
    :return:
    """

    #cgs=[[1.0,'auto']]
    #cgs=[[20.0, 200000.0], [2000.0, 200000.0], [2000.0, 0.002], [0.2, 20.0], [20.0, 20.0], [0.2, 0.2], [200000.0, 0.002], [2000.0, 0.2], [20.0, 0.2]]
    t0=time.time()
    accuracies = []
    pool = Pool()
    fargs = []
    for ndata in priceData:
        fargs.append([ndata, priceData, withArticles,20.0,20.0])
    results = pool.map(trainSCVchild, fargs)
    pool.close()
    pool.join()
    t1=time.time()
    
    for i in results:
        accuracies.append(i[0])
        priceData[i[0][0]]['cum_ret'] = i[1]['cum_ret']
        if withArticles:
            priceData[i[0][0]]['svm_cum_strat_ret_w'] = i[1]['cum_strat_ret']
        else:
            priceData[i[0][0]]['svm_cum_strat_ret_wo'] = i[1]['cum_strat_ret']

    accuracies = sorted(accuracies, key=lambda so: so[2])
    print('Avg test accuracy:', sum(i[2] for i in accuracies) / len(accuracies))
    print("Time: ", t1 - t0)
    for acc in accuracies:
        print("%6s   %.4f   %.4f   %.3f" % (stocks[acc[0]] + ' ('+acc[0]+')', acc[1], acc[2], acc[3]))
    #for acc in accuracies:
    #    print("%6s   %.4f   %.4f   %.3f" % (acc[0], acc[1], acc[2], acc[3]))
    print()

    return priceData


def trainSCVchild(d):
    """
    Child SCV function
    :param d:
    :return:
    """
    ndata = d[0]
    priceData = d[1]
    withArticles=d[2]
    c=d[3]
    g=d[4]
    data = priceData[ndata]
    
    # Calculating future price bool difference
    y = np.where(data['Close'].shift(-1) > data['Close'], 1, -1)
    
    data['Open-Close'] = data.Open - data.Close
    data['High-Low'] = data.High - data.Low
    X = data[['Open-Close', 'High-Low']]
    class_weights=None
    if withArticles:
        X.loc[:, 'Down'] = data.loc[:, 'Down']
        X.loc[:, 'Up'] = data.loc[:, 'Up']
    
    split = int(split_percentage * len(data))
    
    # Train data set
    X_train = X[:split]
    y_train = y[:split]
    
    # Test data set
    X_test = X[split:]
    y_test = y[split:]
    # TODO change tol, manually set gamma and range, range C, ovo vs ovr
    
    sv = SVC(C=c, kernel='rbf', degree=3, gamma=g, coef0=0.0, shrinking=True, \
             probability=True, cache_size=200, class_weight=class_weights, max_iter=-1, \
             decision_function_shape='ovr', random_state=0, tol=0.0000001)
    cls = sv.fit(X_train, y_train)
    
    accuracy_train = accuracy_score(y_train, cls.predict(X_train))
    
    accuracy_test = accuracy_score(y_test, cls.predict(X_test))

    data['Predicted_Signal'] = cls.predict(X)
    
    # Calculate log returns
    data.loc[:, 'Return'] = np.log(data.loc[:, 'Close'].shift(-1) / data.loc[:, 'Close']) * 100
    data.loc[:, 'cum_ret'] = data[split:]['Return'].cumsum()
    
    data.loc[:, 'Strategy_Return'] = data.loc[:, 'Predicted_Signal'] * data.loc[:, 'Return']
    data.loc[:, 'cum_strat_ret'] = data[split:]['Strategy_Return'].cumsum()
    
    std = data['cum_strat_ret'].std()
    Sharpe = (data['cum_strat_ret'] - data['cum_ret']) / std
    Sharpe = Sharpe.mean()
    
    accuracies = [ndata, accuracy_train * 100, accuracy_test * 100, Sharpe]
    

    return accuracies, data


def neuralNet(priceData, withArticles):
    activation = ['identity', 'logistic', 'tanh', 'relu']
    solver = ['lbfgs', 'sgd', 'adam']
    learning_rate = ['constant', 'invscaling', 'adaptive']
    

    #adam was best
    #no identity or sigmoid
    t0=time.time()
    fargs = []
    accuracies = []
    pool = Pool()
    for ndata in priceData:
        fargs.append([ndata, priceData, 'relu', 'adam', 'constant', withArticles])
    results = pool.map(nnetChild, fargs)
    pool.close()
    pool.join()
    for i in results:
        accuracies.append(i[0])
        if withArticles:
            priceData[i[0][0]]['nnet_cum_strat_ret_w'] = i[1]['cum_strat_ret']
        else:
            priceData[i[0][0]]['nnet_cum_strat_ret_wo'] = i[1]['cum_strat_ret']

    accuracies = sorted(accuracies, key=lambda so: so[2])
    print('Avg test accuracy:', sum(i[2] for i in accuracies) / len(accuracies))

    std = []
    for i in accuracies:
        std.append(i[2])
    print(statistics.stdev(std))
    t1=time.time()
    print(t1-t0)
    for acc in accuracies:
        print("%6s   %.4f   %.4f   %.3f" % (stocks[acc[0]] + ' ('+acc[0]+')', acc[1], acc[2], acc[3]))
    #for acc in accuracies:
    #    print("%6s   %.4f   %.4f   %.3f" % (acc[0], acc[1], acc[2], acc[3]))

    print()
    return priceData

def nnetChild(fargs):
    ndata = fargs[0]
    priceData = fargs[1]
    activation = fargs[2]
    solver = fargs[3]
    learning_rate = fargs[4]
    withArticles = fargs[5]
    data = priceData[ndata]
    
    # Calculating future price bool difference
    y = np.where(data['Close'].shift(-1) > data['Close'], 1, -1)
    
    data['Open-Close'] = data.Open - data.Close
    data['High-Low'] = data.High - data.Low
    X = data[['Open-Close', 'High-Low']]
    if withArticles:
        X.loc[:, 'Down'] = data.loc[:, 'Down']
        X.loc[:, 'Up'] = data.loc[:, 'Up']

    # TODO StandardScalar for feature scaling?
    split = int(split_percentage * len(data))
    
    # Train data set
    X_train = X[:split]
    y_train = y[:split]
    
    # Test data set
    X_test = X[split:]
    y_test = y[split:]
    
    mlp = MLPClassifier(hidden_layer_sizes=(100,), activation=activation, solver=solver, alpha=0.0001, batch_size='auto'
                        ,learning_rate=learning_rate,power_t=0.5, max_iter=2000, shuffle=True, random_state=0,
                        tol=0.000001, momentum=0.9, nesterovs_momentum=True, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
                        n_iter_no_change=10, early_stopping=True)
    cls = mlp.fit(X_train, y_train)
    
    accuracy_train = accuracy_score(y_train, cls.predict(X_train))
    
    accuracy_test = accuracy_score(y_test, cls.predict(X_test))
    
    data['Predicted_Signal'] = cls.predict(X)
    
    # Calculate log returns
    data.loc[:, 'Return'] = np.log(data.loc[:, 'Close'].shift(-1) / data.loc[:, 'Close']) * 100
    data.loc[:, 'cum_ret'] = data[split:]['Return'].cumsum()
    
    data.loc[:, 'Strategy_Return'] = data.loc[:, 'Predicted_Signal'] * data.loc[:, 'Return']
    data.loc[:, 'cum_strat_ret'] = data[split:]['Strategy_Return'].cumsum()
    
    std = data['cum_strat_ret'].std()
    Sharpe = (data['cum_strat_ret'] - data['cum_ret']) / std
    Sharpe = Sharpe.mean()
    
    accuracies = [ndata, accuracy_train * 100, accuracy_test * 100, Sharpe]
    
    return accuracies, data
  





def readPriceData(dataconcat):
    priceData = {}
    tpriceData = {}
    dailyData = {}
    for f in os.listdir("./pickles"):
        if 'daily' in f:
            dailyData[f[:-9]] = pd.read_pickle("./pickles/" + f)
        else:  # intraday
            tpriceData[f[:-4]] = pd.read_pickle("./pickles/" + f)
    for index in indices:
        data = pd.DataFrame()
        for interval in dataconcat:
            tdata = tpriceData[index + interval]
            if data.empty:
                data = tdata
            else:
                for i in tdata.index:
                    if i in data.index:
                        data = data[:list(data.index).index(i) + 1]
                        data = data.append(tdata[1:])
                        break
        priceData[index] = data
    for stock in stocks.keys():
        data = pd.DataFrame()
        for interval in dataconcat:
            tdata = tpriceData[stock + interval]
            if data.empty:
                data = tdata
            else:
                for i in tdata.index:
                    if i in data.index:
                        data = data[:list(data.index).index(i) + 1]
                        data = data.append(tdata[1:])
                        break
        priceData[stock] = data
    # TODO 2008 crash
    # if startAfterCrash:
    #   for stock in dailyData:
    
    return priceData, dailyData


def getPriceData():
    """
    Inter-day intervals: (min)
    1 = past 5 days
    5 = past 3 weeks
    15 = past 6 weeks
    30 = past 6 weeks
    60 = past 11 weeks
    :return:
    """
    data = {}
    akey = 0
    intervals = ['daily', '60min', '30min', '15min', '5min', '1min']
    for interval in intervals:
        for index in indices:
            data[index + interval], akey = getAVdata(indices[index], interval, akey)
        for stock in stocks.keys():
            data[stock + interval], akey = getAVdata(stock, interval, akey)
    
    return data


def getAVdata(stock, interval, akey):
    while True:
        try:
            ts = TimeSeries(key=alphakeys[akey], output_format='pandas')
            if interval == 'daily':
                tdata, meta_data = ts.get_daily(symbol=stock, outputsize='full')
            elif interval == 'weekly':
                tdata, meta_data = ts.get_weekly(symbol=stock, outputsize='full')
            elif interval == 'monthly':
                tdata, meta_data = ts.get_monthly(symbol=stock, outputsize='full')
            else:
                tdata, meta_data = ts.get_intraday(symbol=stock, interval=interval, outputsize='full')
            break
        except Exception as e:
            print('Exception!!!: Stock %s   Alphakey: %s  Interval: %s' % (stock, alphakeys[akey], interval))
            akey = (akey + 1) % len(alphakeys)
            print(e, '\n')
            
            continue
    print(stock, alphakeys[akey], interval)
    tdata = tdata.rename(columns={'1. open': 'Open', '2. high': 'High', '3. low': 'Low', \
                                  '4. close': 'Close', '5. volume': 'Volume'})
    tdata = tdata.dropna()
    return tdata, akey

def writeToMatlab(priceData, dailyData, newDailyData):
    print("Writing intraday data to matlab")
    for ndata in priceData:
        with open('matlab/data/' + ndata + '.txt', 'w+') as f:
            data = priceData[ndata]
            data['y'] = np.where(data['Close'].shift(-1) > data['Close'], 1, -1)
            data['Open-Close'] = data.Open - data.Close
            data['High-Low'] = data.High - data.Low
            writeF = ''
            for l in data.index:
                i = data.loc[[l], :]  #:l+1]
                if int(i['y']) == -1:
                    ty = 1
                else:
                    ty = 2
                writeF += '%f,%f,%f,%f,%f,%f,%d\n' % (
                    i['Close'], i['Open'], i['High'], i['Low'], i['Open-Close'], i['High-Low'], ty)
            f.write(writeF)
    print("Writing daily data to matlab")
    for ndata in dailyData:
        with open('matlab/data/' + ndata + '_daily.txt', 'w+') as f:
            data = dailyData[ndata]
            data['y'] = np.where(data['Close'].shift(-1) > data['Close'], 1, -1)
            data['Open-Close'] = data.Open - data.Close
            data['High-Low'] = data.High - data.Low
            writeF = ''
            for l in data.index:
                i = data.loc[[l], :]
                if int(i['y']) == -1:
                    ty = -1
                else:
                    ty = 1
                writeF += '%f,%f,%f,%f,%f,%f,%d\n' % (
                    i['Close'], i['Open'], i['High'], i['Low'], i['Open-Close'], i['High-Low'], ty)
            f.write(writeF)
    print("Writing news article daily data to matlab")
    for ndata in newDailyData:
        with open('matlab/data/' + ndata + '_newDaily.txt', 'w+') as f:
            data = newDailyData[ndata]
            data['y'] = np.where(data['Close'].shift(-1) > data['Close'], 1, -1)
            data['Open-Close'] = data.Open - data.Close
            data['High-Low'] = data.High - data.Low
            writeF = ''
            for l in data.index:
                i = data.loc[[l], :]
                if int(i['y']) == -1:
                    ty = -1
                else:
                    ty = 1
                writeF += '%f,%f,%f,%f,%f,%f,%d,%f,%f\n' % (
                    i['Close'], i['Open'], i['High'], i['Low'], i['Open-Close'], i['High-Low'], ty, i['Down'],
                    i['Up'])
            f.write(writeF)

if __name__ == "__main__":
    main()
