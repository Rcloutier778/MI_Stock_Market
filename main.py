import multiprocessing
from multiprocessing import Pool

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Machine learning classification libraries
from sklearn import preprocessing
from sklearn.svm import SVC, SVR, NuSVC, LinearSVC
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
import tensorflow as tf


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


alphakeys=['R2FFCW41HVNZ8DBN','O6EZ7OAWV5ERVK8S','DVHJN2K8OOWYEO8F','E6K5ZE32ODWBDPOT',\
           'K917CGMUX1MY3NZH','96ZN1WBIGXC6XSOT','3MJILEPY41I90ATY','S2NFK5KCIYCZSVO1',
           'YJKYECKVB9U1XS1Z','W7I10I713Y9PJURK']

##Globals
indices={'DOW':'DJI', 'NASDAQ':'IXIC', 'S&P':'GSPC'}
stocks=['GOOGL','M','BAC','XOM','QQQ','V','DUK','VZ','CVX','PYPL','AMD','JPM']



#Data split percentage for train/test
split_percentage = 0.8

#Get new data
refreshData = False

#Graphs plots
graphPlots = True

def main():
    # From the data intervals, concat the following together to use
    dataconcat = ['30min','15min']#,'5min','1min']#, '5min', '1min']
    tpriceData={}
    priceData={}
    if refreshData:
        #Get and save new data
        tpriceData = getPriceData()
        for i in tpriceData:
            data = tpriceData[i]
            data.to_pickle("pickles/"+i+".pkl")
    else:
        #Use saved data
        #priceData[f] = pd.read_pickle("./pickles/" + f)
        for f in os.listdir("./pickles"):
            tpriceData[f[:-4]] = pd.read_pickle("./pickles/"+f)
        for index in indices:
            data=pd.DataFrame()
            for interval in dataconcat:
                tdata = tpriceData[index+interval]
                if data.empty:
                    data=tdata
                else:
                    for i in tdata.index:
                        if i in data.index:
                            data=data[:list(data.index).index(i)+1]
                            data = data.append(tdata[1:])
                            break
            priceData[index] = data
        for stock in stocks:
            data = pd.DataFrame()
            for interval in dataconcat:
                tdata = tpriceData[stock+interval]
                if data.empty:
                    data=tdata
                else:
                    for i in tdata.index:
                        if i in data.index:
                            data = data[:list(data.index).index(i) + 1]
                            data=data.append(tdata[1:])
                            break
            priceData[stock]=data
    trainPriceData(priceData)



def trainPriceData(priceData):
    trainSVC(priceData)
    #trainTensorFlow(priceData)
    if graphPlots:
        plt.pause(0.001)
        plt.waitforbuttonpress()


def tensorMulti(ndata,priceData):
    ndata = d[0]
    priceData = d[1]
    data = priceData[ndata]

    # data['Return'] = np.log(data['Close'] / data['Close'].shift(1)) * 100
    # y=data['Return'].shift(-1)'
    # Calculating future price bool difference
    y = np.where(data['Close'].shift(-1) > data['Close'], 1, -1)
    # y = np.where(data['Close'].shift(-1) - data['Close'] > 0.0, 1, -1)
    # TODO try labelEncoder in scikit
    # TODO try inverting all y values, compare to non inverted results, use best/most consistent/closest to return
    # lab_enc =preprocessing.LabelEncoder()
    # y = (lab_enc.fit_transform(data['Close'].shift(-1)/data['Close']))
    # y = (((data['Close'].shift(-1) / data['Close']) - np.min(data['Close']))/(np.max(data['Close']) - np.min(data[
    # 'Close'])))

    data['Open-Close'] = data.Open - data.Close
    data['High-Low'] = data.High - data.Low
    X = data[['Close', 'Open-Close', 'High-Low']]
    
    split = int(split_percentage * len(data))
    
    # Train data set
    X_train = X[:split]
    y_train = y[:split]
    
    # Test data set
    X_test = X[split:]
    y_test = y[split:]

    X = tf.placeholder(dtype=tf.float32, shape=[None, n_stocks])
    Y = tf.placeholder(dtype=tf.float32, shape=[None])
    
    # Make Session
    net = tf.Session()
    # Run initializer
    net.run(tf.global_variables_initializer())
    
    # Setup interactive plot
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    line1, = ax1.plot(y_test)
    line2, = ax1.plot(y_test * 0.5)
    plt.show()
    
    # Number of epochs and batch size
    epochs = 10
    batch_size = 256
    n_stocks = 500
    n_neurons_1 = 1024
    n_neurons_2 = 512
    n_neurons_3 = 256
    n_neurons_4 = 128
    n_target = 1
    
    
    # Layer 1: Variables for hidden weights and biases
    W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
    bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
    # Layer 2: Variables for hidden weights and biases
    W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
    bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
    # Layer 3: Variables for hidden weights and biases
    W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
    bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
    # Layer 4: Variables for hidden weights and biases
    W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
    bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))

    # Output layer: Variables for output weights and biases
    W_out = tf.Variable(weight_initializer([n_neurons_4, n_target]))
    bias_out = tf.Variable(bias_initializer([n_target]))
    hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
    hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
    hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
    hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))

    out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))

    mse = tf.reduce_mean(tf.squared_difference(out, Y))

    opt = tf.train.AdamOptimizer().minimize(mse)


    for e in range(epochs):
        
        # Shuffle training data
        shuffle_indices = np.random.permutation(np.arange(len(y_train)))
        X_train = X_train[shuffle_indices]
        y_train = y_train[shuffle_indices]
        
        # Minibatch training
        for i in range(0, len(y_train) // batch_size):
            start = i * batch_size
            batch_x = X_train[start:start + batch_size]
            batch_y = y_train[start:start + batch_size]
            # Run optimizer with batch
            net.run(opt, feed_dict={X: batch_x, Y: batch_y})
            
            # Show progress
            if np.mod(i, 5) == 0:
                # Prediction
                pred = net.run(out, feed_dict={X: X_test})
                line2.set_ydata(pred)
                plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
                file_name = 'img/epoch_' + str(e) + '_batch_' + str(i) + '.jpg'
                plt.savefig(file_name)
                plt.pause(0.01)
    # Print final MSE after Training
    mse_final = net.run(mse, feed_dict={X: X_test, Y: y_test})
    print(mse_final)
    
    
    
    


def trainTensorFlow(priceData):
    accuracies = []
    results=[]
    for ndata in priceData:
        results.append(tensorMulti(ndata, priceData))
    for i in results:
        accuracies.append(i[0])
        priceData[i[0][0]] = i[1]
    
    accuracies = sorted(accuracies, key=lambda so: so[2])
    for acc in accuracies:
        print("%6s   %.4f   %.4f   %.3f" % (acc[0], acc[1], acc[2], acc[3]))
        if graphPlots:
            plt.figure(acc[0], figsize=(10, 5))
            data = priceData[acc[0]]
            data.cum_ret.plot(color='r', label='Returns')
            data.cum_strat_ret.plot(color='g', label='Strategy Returns')
            plt.title(acc[0])
            plt.legend(['Returns', 'Strategy'])
            plt.grid(True)
            plt.show(block=False)
    
    print()
    
    return accuracies


def multiSVC(d):
    ndata = d[0]
    priceData = d[1]
    data = priceData[ndata]

    #data['Return'] = np.log(data['Close'] / data['Close'].shift(1)) * 100
    #y=data['Return'].shift(-1)'
    #Calculating future price bool difference
    y = np.where(data['Close'].shift(-1) > data['Close'], 1, -1)
    #y = np.where(data['Close'].shift(-1) - data['Close'] > 0.0, 1, -1)
    #TODO try labelEncoder in scikit
    #TODO try inverting all y values, compare to non inverted results, use best/most consistent/closest to return
    #lab_enc =preprocessing.LabelEncoder()
    #y = (lab_enc.fit_transform(data['Close'].shift(-1)/data['Close']))
    #y = (((data['Close'].shift(-1) / data['Close']) - np.min(data['Close']))/(np.max(data['Close']) - np.min(data['Close'])))
    #if ndata=='BAC':
    #    print(y)
    data['Open-Close'] = data.Open - data.Close
    data['High-Low'] = data.High - data.Low
    X = data[['Close','Open-Close', 'High-Low']]

    split = int(split_percentage * len(data))

    # Train data set
    X_train = X[:split]
    y_train = y[:split]

    # Test data set
    X_test = X[split:]
    y_test = y[split:]
    #TODO change tol, manually set gamma and range, range C, ovo vs ovr
    sv = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, \
             probability=True, cache_size=200, class_weight=None, max_iter=-1, \
             decision_function_shape='ovr', random_state=None, tol=0.0000001)
    cls = sv.fit(X_train, y_train)


    accuracy_train = accuracy_score(y_train, cls.predict(X_train))
    
    accuracy_test = accuracy_score(y_test, cls.predict(X_test))

    data['Predicted_Signal'] = cls.predict(X)

    # Calculate log returns
    data['Return'] = np.log(data['Close'].shift(-1) / data['Close']) * 100
    data['cum_ret'] = data[split:]['Return'].cumsum()
    if ndata=='JPM':
        print(X)
        for i in range(len(y)):
            print("%2d  %2d"%(y[i],data['Predicted_Signal'][i-1]))
            #print("%2f  %2f"%(data['Close'][i],data['Close'].shift(-1)[i]))
            pass

    data['Strategy_Return'] = data['Predicted_Signal'].shift(-1) * data['Return']
    data['cum_strat_ret'] = data[split:]['Strategy_Return'].cumsum()

    std = data.cum_strat_ret.std()
    Sharpe = (data.cum_strat_ret - data.cum_ret) / std
    Sharpe = Sharpe.mean()

    accuracies=[ndata, accuracy_train * 100, accuracy_test * 100, Sharpe]


    if ndata == 'JPM':
        print("%4s  %12s  %12s  %12s  %12s  %12s  %12s  %7s  %12s  %12s" % ("Num","Open", "Close","High","Low", "Open-Close", "High-Low", "predict", "Return", "Strat Return"))
        for i in range(-40,-20,1):
            print("%4d  %12f  %12f  %12f  %12f  %12f  %12f  %7d  %12f  %12f" % (i, data[i:i+1]['Open'],data[i:i+1]['Close'],data[i:i+1]['High'],data[i:i+1]['Low'],data[i:i+1]['Open-Close'], X[i:i+1]['High-Low'],data['Predicted_Signal'][i:i+1],data.Return[i:i+1],data.Strategy_Return[i:i+1]))
    if graphPlots:
        return accuracies, data
    else:
        return accuracies,0


def trainSVC(priceData):

    accuracies=[]
    pool=Pool()
    fargs=[]

    for ndata in priceData:
        fargs.append([ndata,priceData])
    results = pool.map(multiSVC,fargs)
    pool.close()
    pool.join()
    for i in results:
        accuracies.append(i[0])
        priceData[i[0][0]]=i[1]

    accuracies = sorted(accuracies,key=lambda so: so[2])
    for acc in accuracies:
        print("%6s   %.4f   %.4f   %.3f" % (acc[0], acc[1], acc[2], acc[3]))
        if graphPlots:
            plt.figure(acc[0],figsize=(10, 5))
            data = priceData[acc[0]]
            data.cum_ret.plot(color='r',label='Returns')
            data.cum_strat_ret.plot(color='g',label='Strategy Returns')
            plt.title(acc[0])
            plt.legend(['Returns','Strategy'])
            plt.grid(True)
            plt.show(block=False)


    print()

    return accuracies




    
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
    data={}
    akey=0
    intervals=['60min','30min','15min','5min','1min']
    for interval in intervals:
        for index in indices:
            data[index+interval],akey=getAVdata(indices[index],interval,akey)
        for stock in stocks:
            data[stock+interval],akey=getAVdata(stock,interval,akey)

    return data

def getAVdata(stock,interval,akey):
    while True:
        try:
            ts = TimeSeries(key=alphakeys[akey], output_format='pandas')
            tdata, meta_data = ts.get_intraday(symbol=stock, interval=interval, outputsize='full')
            break
        except ValueError as e:
            print(alphakeys[akey])
            akey = (akey+1) % len(alphakeys)
            print(e)

            continue
    print(stock, alphakeys[akey])
    tdata = tdata.rename(columns={'1. open': 'Open', '2. high': 'High', '3. low': 'Low', \
                                  '4. close': 'Close', '5. volume': 'Volume'})
    tdata = tdata.dropna()
    return tdata,akey




if __name__ == "__main__":
    main()
