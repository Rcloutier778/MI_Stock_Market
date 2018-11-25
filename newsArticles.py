import warnings
from multiprocessing import Pool

warnings.filterwarnings("ignore", category=DeprecationWarning)
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
from datetime import datetime, timedelta

# Machine learning classification libraries
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from  sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif, SelectFwe, SelectKBest, SelectFromModel, SelectPercentile, SelectFdr, SelectFpr, GenericUnivariateSelect
from sklearn.feature_selection.rfe import RFE, RFECV
from sklearn.linear_model import LassoCV

# For data manipulation
import pandas as pd
import numpy as np

# To plot

# To fetch data
import time
import os
from bs4 import BeautifulSoup
import requests
import re
import Summarizer
import json
from collections import defaultdict
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys



refreshArticles = False #Wipe data and start again
updateArticles = False #Update current data

numPages=100 #Number of pages of articles to use
minAccuracy=30 #Minimum accuracy

def getArticles(priceData, dailyData, stocks):
    """
    Main func for all news article stuff
    ;param priceData: intraday data
    :param dailyData: daily price data for all relevent stocks, same format as intraday
    :param stocks: List of stocks
    :return:
    """
    print("Ripping and summariing articles")
    alldata = ripArticles(stocks)  # Get articles and summarize and keywords

    print("Classifying keywords")
    fargs = []
    pool = Pool()

    for stockName in stocks.keys():
        if alldata[stocks[stockName]][0]:
            fargs.append([alldata[stocks[stockName]][0], dailyData[stockName], stockName])

    results = pool.map(parseArticles, fargs)
    pool.close()
    pool.join()

    adata = pd.DataFrame()
    fittedAdata = pd.DataFrame()
    y = []
    countedAllWords = defaultdict(int)
    dataDict = {}
    accuracyDict=defaultdict(dict)

    for result in results:
        tacc = result
        if tacc==None:
            continue
        dataDict[tacc['name']] = tacc
        adata = adata.append(tacc['adata'], ignore_index=True, sort=False)
        if 'adata_new' in tacc.keys():
            fittedAdata = fittedAdata.append(tacc['adata_new'], ignore_index=True, sort=False)
        else:
            fittedAdata = fittedAdata.append(tacc['adata'], ignore_index=True, sort=False)
        accuracyDict[tacc['name']]['stand_alone'] = tacc['accuracy']

        y.extend(tacc['y'])
        for word in tacc['countedAllWords']:
            countedAllWords[word]+=tacc['countedAllWords'][word]

    print("Classifying total words")

    #setup X and y
    adata = adata.fillna(0)
    y = np.array(y)
    fittedAdata = fittedAdata.fillna(0)

    print("Skipping overall accuracy measurments for testing ")
    if False:
        print(len(countedAllWords))
        print(adata.shape)


        #SVM with all words
        accuracies=articleSVM(adata,y,'Overall Accuracies')
        print("%s:  %f  %f" % (accuracies['name'],accuracies['train'],accuracies['test']))

        fittedAccuracies = articleSVM(fittedAdata,y,'Fitted Overall Accuracies')
        print("%s:  %f  %f" % (fittedAccuracies['name'],fittedAccuracies['train'],fittedAccuracies['test']))


    #Most common reoccuring words
    sorted_keys_by_values = sorted(countedAllWords.keys(),key=lambda kv: countedAllWords[kv], reverse=True)
    cutIndex=len(sorted_keys_by_values)//4
    chopped_keys=sorted_keys_by_values[:cutIndex]

    chopped_adata=adata[chopped_keys]
    #SVM with most common words
    chopped_accuracies=articleSVM(chopped_adata,y,'Chopped Overall Accuracies')
    print("%s:  %f  %f" % (chopped_accuracies['name'], chopped_accuracies['train'], chopped_accuracies['test']))

    reducedDimResults=dimReducer(adata,y, multiThread=True)
    reducedDimResults.append(chopped_accuracies)
    reducedDimResults = sorted(reducedDimResults, key= lambda kv: kv['test'], reverse=True)

    for reducedResult in reducedDimResults:
        print("%s: %f  %f" %(reducedResult['name'], reducedResult['train'], reducedResult['test']))
    if 'adata_new' in reducedDimResults[0].keys():
        total_best_fit_adata = reducedDimResults[0]['adata_new']
    else:
        total_best_fit_adata = reducedDimResults[0]['adata']
    total_best_fit_cls = reducedDimResults[0]['cls']

    for stock in dataDict:
        temp_adata = pd.DataFrame(columns=total_best_fit_adata.columns)
        temp_adata = temp_adata.append(dataDict[stock]['adata'], sort=False)
        for i in temp_adata.columns:
            if i not in total_best_fit_adata.columns:
                temp_adata = temp_adata.drop(columns=i)

        temp_adata = temp_adata.fillna(0)
        stock_on_total = accuracy_score(dataDict[stock]['y'], total_best_fit_cls.predict(temp_adata)) *100
        accuracyDict[stock]['total_best_fit_cls'] = stock_on_total
        print("%s on total_best_fit_cls:  %f" % (stock, stock_on_total))
        if 'adata_new' in dataDict[stock].keys():
            temp_adata = pd.DataFrame(columns=total_best_fit_adata.columns)
            temp_adata = temp_adata.append(dataDict[stock]['adata_new'], sort=False)
            for i in temp_adata.columns:
                if i not in total_best_fit_adata.columns:
                    temp_adata = temp_adata.drop(columns=i)

            temp_adata = temp_adata.fillna(0)
            stock_on_total = accuracy_score(dataDict[stock]['y'], total_best_fit_cls.predict(temp_adata)) * 100
            accuracyDict[stock]['total_best_fit_cls reduced'] = stock_on_total
            print("%s on total_best_fit_cls with reduced adata:  %f" % (stock, stock_on_total))





    #TODO using fitted adata below
    print("Using adata_new as adata below:::")

    reducedDimResults = dimReducer(fittedAdata, y, multiThread=True)
    reducedDimResults = sorted(reducedDimResults, key=lambda kv: kv['test'], reverse=True)

    for reducedResult in reducedDimResults:
        print("%s: %f  %f" % (reducedResult['name'], reducedResult['train'], reducedResult['test']))
    if 'adata_new' in reducedDimResults[0].keys():
        total_best_fit_adata = reducedDimResults[0]['adata_new']
    else:
        total_best_fit_adata = reducedDimResults[0]['adata']
    total_best_fit_cls = reducedDimResults[0]['cls']

    for stock in dataDict:
        temp_adata = pd.DataFrame(columns=total_best_fit_adata.columns)
        temp_adata = temp_adata.append(dataDict[stock]['adata'], sort=False)
        for i in temp_adata.columns:
            if i not in total_best_fit_adata.columns:
                temp_adata = temp_adata.drop(columns=i)

        temp_adata = temp_adata.fillna(0)
        stock_on_total = accuracy_score(dataDict[stock]['y'], total_best_fit_cls.predict(temp_adata)) * 100
        accuracyDict[stock]['total_best_fit_cls fitted'] = stock_on_total
        print("%s on total_best_fit_cls:  %f" % (stock, stock_on_total))
        if 'adata_new' in dataDict[stock].keys():
            temp_adata = pd.DataFrame(columns=total_best_fit_adata.columns)
            temp_adata = temp_adata.append(dataDict[stock]['adata_new'], sort=False)
            for i in temp_adata.columns:
                if i not in total_best_fit_adata.columns:
                    temp_adata = temp_adata.drop(columns=i)

            temp_adata = temp_adata.fillna(0)

            stock_on_total = accuracy_score(dataDict[stock]['y'],
                                            total_best_fit_cls.predict(temp_adata)) * 100
            accuracyDict[stock]['total_best_fit_cls reduced fitted'] = stock_on_total
            print("%s on total_best_fit_cls with reduced adata:  %f" % (stock, stock_on_total))

    for stock in accuracyDict.keys():
        sorted_accuracies = sorted(accuracyDict[stock].items(), key=lambda kv: kv[1], reverse=True)
        print(stock)
        for i in sorted_accuracies:
            print("%s:   %f"%(i[0], i[1]))

    #TODO endof fitted adata



    print("Using None for rs in articlSVM!!!!!!")



    #TODO Feature Extraction
    #TODO Dimensionality Reduction

    # TODO return_estimator


    return

def dimReducer(adata, y, multiThread=False):
    """
    Uses feature selection to reduce the number of dimensions  used.
    Multi-threaded
    :param adata:
    :param y:
    :return:
    """
    selections = ['k_best', 'fpr', 'fdr', 'fwe', 'percentile']

    fargs = []
    for select in selections:
        fargs.append([adata, y, select, 'chi2'])

    if multiThread:
        pool = Pool()
        results = pool.map(dimReducerChild, fargs)
        pool.close()
        pool.join()
    else:
        results=[]
        for i in fargs:
            results.append(dimReducerChild(i))
    return results

def dimReducerChild(fargs):
    adata = fargs[0]
    y = fargs[1]
    select = fargs[2]
    classm = fargs[3]
    additionalParams={'k_best':200, 'percentile':10, 'fpr':0.005, 'fdr':0.005, 'fwe':0.005}
    adata_new = GenericUnivariateSelect(score_func=eval(classm), mode=select, param=additionalParams[select]).fit_transform(adata, y)
    accuracies = articleSVM(adata_new,y,select + " " + classm)
    accuracies['adata_new']=pd.DataFrame(data=adata_new)
    return accuracies


def ripArticles(stocks):
    """
    Go to NYTimes, search for company name, summarize articles and place in dict
    with key being article publish date. Returns dict of summarized articles.
    """

    #Get new article links
    if refreshArticles:
        articleLinks = getArticleLinks(list(stocks.values()))
        with open('summarizedArticles/articleLinks.json', 'w+') as f:
            json.dump(articleLinks, f, separators=(',', ':'))
        print("Saved article links")
    else:
        if os.path.isfile('summarizedArticles/articleLinks.json'):
            with open('summarizedArticles/articleLinks.json','r') as f:
                articleLinks = json.load(f)
            print("Loaded article links")
        else:
            articleLinks = getArticleLinks(list(stocks.values()))
            with open('summarizedArticles/articleLinks.json', 'w+') as f:
                json.dump(articleLinks, f, separators=(',', ':'))
            print("Saved article links")

    pool = Pool()
    alldata = {}
    fargs = []

    print("Reading articles")
    for stockName in stocks.values():
        fargs.append([stockName, articleLinks[stockName], refreshArticles, updateArticles])
    results = pool.map(ripArticlesChild, fargs)
    pool.close()
    pool.join()
    for i in results:
        alldata[i[0]] = i[1]
    print("Finished ripping articles")
    return alldata



def ripArticlesChild(fargs):
    """
    Search NYTimes for a company.
    Read all applicable articles.
    Summarize all articles.

    """
    stockName = fargs[0]
    links = fargs[1]
    refArt = fargs[2] #Clear out saved articles and summarize from scratch
    upArt = fargs[3] #Add new articles to saved

    # https://github.com/miso-belica/sumy
    # XPATH of parent container of article body=    //*[@id="story"]/section
    # XPATH of first body subsection=     //*[@id="story"]/section/div[1]
    quote_page = 'https://www.nytimes.com/'
    qpage = 'https://www.nytimes.com/search?query='

    if refArt:
        # If resetting article data
        data = [defaultdict(list), []]
    else:
        if os.path.isfile('summarizedArticles/' + stockName + '.json'):
            with open('summarizedArticles/' + stockName + '.json','r') as f:
                # [ {datetime:[summarized article]}, [included urls] ]
                # [ dict of lists, list ]
                data = json.load(f)
            if not upArt:
                #If not updating the articles
                return stockName, data
        else:
            data = [defaultdict(list), []]

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
            try:
                datetime.strptime(articleDate, "%Y-%m-%d")
            except ValueError:
                try:
                    articleDate = articleDate.replace("Sept", "Sep")
                    dt = datetime.strptime(articleDate, "%b. %d, %Y")
                    articleDate = dt.strftime("%Y-%m-%d")
                except ValueError:
                    #Has UTC offset
                    dt = datetime.strptime(articleDate, "%Y-%m-%dT%H:%M:%S%z")
                    articleDate = dt.strftime("%Y-%m-%d")
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

    with open('summarizedArticles/' + stockName + '.json', 'w+') as f:
        json.dump(data, f, separators=(',', ':'))

    print("Finished %s" % stockName)
    return stockName, data


def getArticleLinks(names):
    """
    Gets links to articles associated with a company. Uses Selenium.
    :param names: list of human readable names of stocks
    :return: Dict of company names to lists of associated articles
    """
    print("Getting article links")
    driver = webdriver.Chrome()
    qpage = 'https://www.nytimes.com/search?query='
    quote_page = 'https://www.nytimes.com/'
    linkDict = {}
    for name in names:
        print(name)
        spage = qpage + name.replace(' ', '%20')
        driver.get(spage)
        time.sleep(3)
        ActionChains(driver).key_down(Keys.PAGE_DOWN)
        ActionChains(driver).send_keys(Keys.PAGE_DOWN)
        time.sleep(1)
        ActionChains(driver).key_up(Keys.PAGE_DOWN)
        time.sleep(1)

        button = None

        # Try to find the xpath to the Show More button element
        for lcv in range(20):
            try:
                parentButtonXpaths = driver.find_elements_by_xpath(""".//*[@id="site-content"]/div/div/div[2]/*""")
                button = driver.find_element_by_xpath("""//*[@id="site-content"]/div/div/div[2]/div[""" + str(
                    len(parentButtonXpaths)) + """]/div/button""")
                break
            except:
                time.sleep(1)
                pass

        if not button:
            # If button was not found, go to next stock
            continue

        # Scroll the page down
        ActionChains(driver).key_down(Keys.PAGE_DOWN)
        time.sleep(0.5)
        for i in range(numPages):
            clicked = False
            for lcv in range(7):
                try:
                    button.click()
                    time.sleep(0.5)
                    clicked = True
                    break
                except:
                    time.sleep(1)
                    pass
            if not clicked:
                break
        ActionChains(driver).key_up(Keys.PAGE_DOWN)

        # Get the page html source for beautiful soup parsing
        r = driver.page_source
        soup = BeautifulSoup(r, 'html.parser')
        reg = re.compile('.*SearchResults-item.*')
        f = soup.find_all('li', attrs={'class': reg})
        links = []
        # Get links from search
        for i in f:
            reg = re.compile('.*Item-section--.*')
            try:
                section = i.find('p', attrs={'class': reg}).text.lower()

                if section in ['technology', 'business', 'climate', 'energy & environment']:
                    links.append(quote_page + i.find('a').get('href'))
            except:
                pass
        linkDict[name] = links
        print("Number of links: %d"%len(links))

    # Gracefully close driver
    driver.close()
    driver.quit()

    return linkDict


def parseArticles(fargs):
    """
    Multi-threaded article parser.
    Finds the effects of a news article on the company it's based on.
    Looks ahead 1, 3, and 5 days to see lasting effects.
    :param fargs: List of [dict of datetime -- 2D list of articles and keywords,
                                daily price data,
                                stock name]
    :return: List of [dataframe of keyword data (number of occurances of keyword per look ahead impact date,
                        ground truth sotck price rise/falls,
                        dict of {words : # of occurances}]
    """
    articleDict = fargs[0]
    dailyData = fargs[1]
    stockName = fargs[2]
    # Dict of article keywords and occurances sorted by publish date of  article
    # Daily price data

    #All words with appearance count
    countedAllWords = defaultdict(int)
    for articleDate in articleDict:
        for article in articleDict[articleDate]:
            for kw in article.keys():
                countedAllWords[kw] += article[kw]

    #All words
    allwords = list(countedAllWords.keys())

    # Matrix. Rows=days, Cols=words mentioned
    adata = []

    # -1,0,1 representing changes in price
    y = []

    for articlesDate in articleDict:
        dt = datetime.strptime(articlesDate, '%Y-%m-%d')

        day0 = (dt).strftime("%Y-%m-%d")
        day1 = (dt + timedelta(days=1)).strftime("%Y-%m-%d")
        day3 = (dt + timedelta(days=3)).strftime("%Y-%m-%d")
        day5 = (dt + timedelta(days=5)).strftime("%Y-%m-%d")
        days = [day0, day1, day3, day5]
        dayIndex = []
        for day in days:
            if day in dailyData.index.values:
                dayIndex.append(dailyData.index.get_loc(day))

        # Price direction for day+1,3,5
        dayDiff = []
        if len(dayIndex) > 2:
            for index in range(1, len(dayIndex)):
                diff = dailyData['Close'][dayIndex[index]] - dailyData['Open'][dayIndex[0]]
                if diff > 0:
                    dayDiff.append(1)
                elif diff < 0:
                    dayDiff.append(-1)
                else:
                    dayDiff.append(0)
        else:
            # If article is too recent, disregard it
            continue

        articles = articleDict[articlesDate]
        if len(articles) > 1:
            temp_article = defaultdict(int)
            for a in articles:
                for b in a:
                    temp_article[b] += a[b]
            articles = temp_article
        else:
            articles = articles[0]
        for day in range(0, len(dayDiff)):
            arow = []
            for word in allwords:
                if word in articles.keys():
                    arow.append(articles[word])
                else:
                    arow.append(0)
            adata.append(arow)
            y.append(dayDiff[day])

    adata = pd.DataFrame(np.array(adata), columns=allwords)
    y = np.array(y)

    # Return if all outputs are the same
    if all(y[0] == ytemp for ytemp in y):
        print("%s All outputs the same" % stockName)
        return 0, 0, 0
    if adata.shape[1] > 700:
        print(adata.shape)
        accuracies=dimReducer(adata,y)
        accuracies=sorted(accuracies, key=lambda kv : kv['test'], reverse=True)[0]
        if 'name' not in accuracies:
            raise("Error occured when trying to reduce dimensionality. Figure something out.")
        accuracies['name']=stockName + " reduced with " + accuracies['name']
    else:
        accuracies=articleSVM(adata,y,stockName)

    if accuracies['test'] < minAccuracy:
        if adata.shape[1]  >  700:
            accuracies=articleSVM(adata,y,stockName)
            if accuracies['test'] < minAccuracy:
                # If accuracy is less than minimum
                print("%s had less than acceptable accuracy. %f " % (stockName, accuracies['test']))
                return None
        else:
            #If accuracy is less than minimum
            print("%s had less than acceptable accuracy. %f " % (stockName, accuracies['test']))
            return None
    print("%s:  %f  %f" % (accuracies['name'], accuracies['train'], accuracies['test']))

    ret = {'adata':adata, 'y':y, 'countedAllWords':countedAllWords, 'name':stockName,
                'accuracy':accuracies['test'], 'cls':accuracies['cls']}

    if 'adata_new' in accuracies:
        #Add the reduced adata to return
        if type(accuracies['adata_new']) == np.ndarray:
            ret['adata_new'] = pd.DataFrame(data=accuracies['adata_new'])
        else:
            ret['adata_new'] = accuracies['adata_new']

    return ret


def articleSVM(X,y, name):
    """
    Classify the X data with SVM.
    Use 20% test set size for cross validation.
        Sets cross-validation random state to a pre-defined number
    :param X: Data matrix
    :param y: Ground truth answers
    :param name: Name of what the data in X is representing
    :return: List of [name, training accuracy, testing accuracy]
    """
    y_train = [0, 0]
    rs = None   # Used to prevent infinite loop
    while all(y_train[0] == ytemp for ytemp in y_train):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rs)
        #rs += 1

    sv = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, \
             probability=True, cache_size=200, class_weight=None, max_iter=-1, \
             decision_function_shape='ovr', random_state=None, tol=0.00001)

    cls = sv.fit(X_train, y_train)

    accuracy_train = accuracy_score(y_train, cls.predict(X_train))

    accuracy_test = accuracy_score(y_test, cls.predict(X_test))

    accuracies = {'name':name, 'train':accuracy_train * 100, 'test':accuracy_test * 100, 'cls': cls}
    return accuracies
