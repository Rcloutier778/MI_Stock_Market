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
from bs4 import BeautifulSoup
import requests
import re
import urllib3.request
import Summarizer
import json
from collections import defaultdict, OrderedDict, Counter


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
indices={'DOW':'DJI', 'NASDAQ':'IXIC'}
#Relate Jpmorgan and chase bank?
stocks={'GOOGL':'Google','M':'Macys','BAC':'Bank of America','XOM':'Exxon Mobil',
            'QQQ':'PowerShares QQQ Trust','V':'Visa','DUK':'Duke Energy',
            'VZ':'Verizon','CVX':'Chevron','PYPL':'Paypal','AMD':'AMD',
            'JPM':'J.P. Morgan'}



#Data split percentage for train/test
split_percentage = 0.8

#Get new data
refreshData = False

#Graphs plots
graphPlots = True

def main():
    # From the data intervals, concat the following together to use
    if refreshData:
        #Get and save new data
        priceData = getPriceData()
        for i in priceData:
            data = priceData[i]
            data.to_pickle("pickles/"+i+".pkl")

    dataconcat = ['30min','15min']#,'5min','1min']#, '5min', '1min']
    #Use saved data
    priceData,dailyData=readPriceData(dataconcat)

    #@@@for matlab, delete later
    matlab=False
    if matlab:
        for ndata in priceData:
            with open('matlab/data/'+ndata+'.txt', 'w+') as f:
                data=priceData[ndata]
                data['y'] = np.where(data['Close'].shift(-1) > data['Close'], 1, -1)
                data['Open-Close'] = data.Open - data.Close
                data['High-Low'] = data.High - data.Low
                for l in range(779):
                    i=data[l:l+1]
                    if int(i['y']) ==-1:
                        ty=1
                    else:
                        ty=2
                    f.write('%f,%f,%f,%f,%f,%f,%d\n'%(i['Close'],i['Open'],i['High'],i['Low'],i['Open-Close'],i['High-Low'],ty))
    
        return

    #@@@end for matlab


    getArticles(dailyData)

    #Intraday data prediction
    trainSVC(priceData)
    # trainTensorFlow(priceData)
    if graphPlots:
        plt.pause(0.001)
        plt.waitforbuttonpress()

    #Daily data prediction
    trainSVC(dailyData)
    # trainTensorFlow(priceData)
    if graphPlots:
        plt.pause(0.001)
        plt.waitforbuttonpress()


def parseArticles(fargs):
    #Dict of article keywords and occurances sorted by publish date of  article
    articleDict=fargs[0]
    #Daily price data
    dailyData=fargs[1]

    #Set of all keywords found across all articles of a company
    allwords=set()
    for articleDate in articleDict:
        for article in articleDict[articleDate]:
            for kw in article.keys():
                allwords.add(kw)
    
    allwords=list(allwords)
    
    #Matrix. Rows=days, Cols=words mentioned
    adata=[]

    #-1,0,1 representing changes in price 
    y=[]
    
    
    for articlesDate in articleDict:
        day0 = articlesDate
        dt = datetime.strptime(day0,'%Y-%m-%d')
        day1 = (dt + timedelta(days=1)).strftime("%Y-%m-%d")
        day3 = (dt + timedelta(days=3)).strftime("%Y-%m-%d")
        day5 = (dt + timedelta(days=5)).strftime("%Y-%m-%d")
        days=[day0,day1,day3,day5]
        dayIndex=[]
        for day in days:
            if day in dailyData.index.values:
                dayIndex.append(dailyData.index.get_loc(day))

        #Price direction for day+1,3,5
        dayDiff=[]
        if len(dayIndex) > 2:
            for index in range(1,len(dayIndex)):
                diff=dailyData['Close'][dayIndex[index]]-dailyData['Open'][dayIndex[0]]
                if diff>0:
                    dayDiff.append(1)
                elif diff<0:
                    dayDiff.append(-1)
                else:
                    dayDiff.append(0)
        else:
            #If article is too recent, disregard it
            continue

        articles=articleDict[articlesDate]
        if len(articles) > 1:
            temp_article=defaultdict(int)
            for a in articles:
                for b in a:
                    temp_article[b] += a[b]
            articles=temp_article
        else:
            articles=articles[0]
        for day in range(0,len(dayDiff)):
            arow=[]
            for word in allwords:
                if word in articles.keys():
                    arow.append(articles[word])
                else:
                    arow.append(0)
            adata.append(arow)
            y.append(dayDiff[day])
    
    adata = pd.DataFrame(np.array(adata), columns=allwords)    
    y = np.array(y)

    split = int(split_percentage * len(adata))

    # Train data set
    X_train = adata[:split]
    y_train = y[:split]

    # Test data set
    X_test = adata[split:]
    y_test = y[split:]
    
    #TODO change tol, manually set gamma and range, range C, ovo vs ovr
    sv = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, \
             probability=True, cache_size=200, class_weight=None, max_iter=-1, \
             decision_function_shape='ovr', random_state=None, tol=0.001)
    try:
        cls = sv.fit(X_train, y_train)
    except Exception as e:
        print(X_train)
        print(y_train)
        raise(e)

    accuracy_train = accuracy_score(y_train, cls.predict(X_train))
    
    accuracy_test = accuracy_score(y_test, cls.predict(X_test))

    accuracies=[accuracy_train * 100, accuracy_test * 100]
    print(accuracies)
    
    return wordWeight


def getArticles(priceData):
    """
    main func for articles
    """
    alldata = ripArticles() #Get articles and summarize and keywords
    pool = Pool()
    fargs = []
    wordWeights = defaultdict(int)
    
    #Multi-thread word weights
    for stockName in stocks.keys():
        fargs.append([alldata[stocks[stockName]][0],priceData[stockName]])










    results = pool.map(parseArticles, fargs)
    pool.close()
    pool.join()
    for i in results:
        for word in i:
            wordWeights[word] += i[word]

    print(wordWeights)


    1/0
    return

def ripArticlesChild(stockName):
    """
    Search NYTimes for a company. 
    Read all applicable articles. 
    Summarize all articles.
    
    """
    # https://github.com/miso-belica/sumy
    # XPATH of parent container of article body=    //*[@id="story"]/section
    # XPATH of first body subsection=     //*[@id="story"]/section/div[1]
    quote_page = 'https://www.nytimes.com/'
    qpage = 'https://www.nytimes.com/search?query='

    if os.path.isfile('summarizedArticles/'+stockName + '.json'):
        with open('summarizedArticles/'+stockName + '.json') as f:
            # [ {datetime:[summarized article]}, [included urls] ]
            # [ dict of lists, list ]
            data = json.load(f)

    else:
        data = [defaultdict(list), []]
    spage = qpage + stockName.replace(' ', '%20')
    r = requests.get(spage)
    assert r.status_code == 200
    soup = BeautifulSoup(r.content, 'html.parser')
    reg = re.compile('.*SearchResults-item.*')
    f = soup.find_all('li', attrs={'class': reg})
    links = []
    for i in f:
        reg = re.compile('.*Item-section--.*')
        try:
            section = i.find('p', attrs={'class': reg}).text.lower()

            if section in ['technology', 'business','climate','energy & environment']:
                links.append(quote_page + i.find('a').get('href'))
        except:
            pass

    for link in links:
        if link in data[1]:
            continue
        data[1].append(link)
        r = requests.get(link)
        soup = BeautifulSoup(r.content, 'html.parser')
        reg = re.compile('.*StoryBodyCompanionColumn.*')
        f = soup.find_all('div', attrs={'class': reg})
        text = ''
        if soup.find('time'):
            articleDate = soup.find('time')['datetime']
        else:
            # Not an article
            continue
        # Get date and time of publication

        for k in f:  # for each BodyCompanionColumn
            k = k.find_all('p')  # Find all paragraphs
            for p in k:  # for each paragraph
                # get text
                text += p.text

        summarizedText = Summarizer.getSummary(text.strip())
        keywords = Summarizer.getKeywords(summarizedText)
        if articleDate in data[0].keys():
            data[0][articleDate].append(keywords)
        else:
            data[0][articleDate] = [keywords]

    with open('summarizedArticles/'+stockName + '.json', 'w+') as f:
        json.dump(data, f, separators=(',', ':'))

    return stockName,data



def ripArticles():
    """
    Go to NYTimes, search for company name, summarize articles and place in dict
    with key being article publish date. Returns dict of summarized articles.
    """
    pool=Pool()
    alldata={}
    fargs = []
    for stockName in stocks.values():
        fargs.append(stockName)
    results = pool.map(ripArticlesChild,fargs)
    pool.close()
    pool.join()
    for i in results:
        alldata[i[0]] = i[1]

    return alldata

        
def multiSVC(d):
    ndata = d[0]
    priceData = d[1]
    data = priceData[ndata]

    #Calculating future price bool difference
    y = np.where(data['Close'].shift(-1) > data['Close'], 1, -1)
    
    #TODO try labelEncoder in scikit
    #TODO try inverting all y values, compare to non inverted results, use best/most consistent/closest to return
    #lab_enc =preprocessing.LabelEncoder()
    #y = (lab_enc.fit_transform(data['Close'].shift(-1)/data['Close']))
    #y = (((data['Close'].shift(-1) / data['Close']) - np.min(data['Close']))/(np.max(data['Close']) - np.min(data['Close'])))

    data['Open-Close'] = data.Open - data.Close
    data['High-Low'] = data.High - data.Low
    X = data[['Open-Close','High-Low']]

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

    data['Strategy_Return'] = (data['Predicted_Signal']) * data['Return']
    data['cum_strat_ret'] = data[split:]['Strategy_Return'].cumsum()

    std = data.cum_strat_ret.std()
    Sharpe = (data.cum_strat_ret - data.cum_ret) / std
    Sharpe = Sharpe.mean()

    accuracies=[ndata, accuracy_train * 100, accuracy_test * 100, Sharpe]

    '''
    if ndata == 'JPM':
        print("%4s  %12s  %12s  %12s  %12s  %12s  %12s  %7s  %12s  %12s" % ("Num","Open", "Close","High","Low", "Open-Close", "High-Low", "predict", "Return", "Strat Return"))
        for i in range(-40,-20,1):
            print("%4d  %12f  %12f  %12f  %12f  %12f  %12f  %7d  %12f  %12f" % (i, data[i:i+1]['Open'],data[i:i+1]['Close'],data[i:i+1]['High'],data[i:i+1]['Low'],data[i:i+1]['Open-Close'], X[i:i+1]['High-Low'],data['Predicted_Signal'][i:i+1],data.Return[i:i+1],data.Strategy_Return[i:i+1]))'''
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
    print('Avg test accuracy:',sum(i[2] for i in accuracies)/len(accuracies))
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


def readPriceData(dataconcat):
    priceData={}
    tpriceData={}
    dailyData={}
    for f in os.listdir("./pickles"):
        if 'daily' in f:
            dailyData[f[:-9]]= pd.read_pickle("./pickles/" + f)
        else: #intraday
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
    data={}
    akey=0
    intervals=['daily','60min','30min','15min','5min','1min']
    for interval in intervals:
        for index in indices:
            data[index+interval],akey=getAVdata(indices[index],interval,akey)
        for stock in stocks.keys():
            data[stock+interval],akey=getAVdata(stock,interval,akey)

    return data


def getAVdata(stock,interval,akey):
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
            print('Exception!!!: Stock %s   Alphakey: %s  Interval: %s' % (stock, alphakeys[akey],interval))
            akey = (akey+1) % len(alphakeys)
            print(e,'\n')

            continue
    print(stock, alphakeys[akey], interval)
    tdata = tdata.rename(columns={'1. open': 'Open', '2. high': 'High', '3. low': 'Low', \
                                  '4. close': 'Close', '5. volume': 'Volume'})
    tdata = tdata.dropna()
    return tdata,akey




if __name__ == "__main__":
    main()
