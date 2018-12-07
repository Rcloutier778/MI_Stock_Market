import warnings
from multiprocessing import Pool

warnings.filterwarnings("ignore", category=DeprecationWarning)
import matplotlib.cbook
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore",category=matplotlib.MatplotlibDeprecationWarning)
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
from datetime import datetime, timedelta, date

# Machine learning classification libraries
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif, SelectFwe, SelectKBest, SelectFromModel, \
    SelectPercentile, SelectFdr, SelectFpr, GenericUnivariateSelect
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
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import pickle
import copy
pd.options.mode.chained_assignment = None

refreshArticles = False  # Wipe data and start again
updateArticles = False  # Update current data
refreshModels = False
cutoff = False

numPages = 100  # Number of pages of articles to use
minAccuracy = 30  # Minimum accuracy


def getArticles(priceData, dailyData, stocks):
    """
    Main func for all news article stuff
    ;param priceData: intraday data
    :param dailyData: daily price data for all relevent stocks, same format as intraday
    :param stocks: List of stocks
    :return:
    """
    pd.options.mode.chained_assignment = None
    modelsDict = {}
    if refreshModels:
        modelsDict, modelsColsDict = getArticleModels(priceData, dailyData, stocks)
    else:
        for stock in stocks:
            if os.path.isfile('models/' + stock + '.pkl'):
                modelsDict[stock] = pickle.load(open('models/' + stock + '.pkl', 'rb'))
            else:
                print("%s does not have a saved model" % stock)
        if os.path.isfile('models/overall.pkl'):
            overallModel = pickle.load(open('models/overall.pkl', 'rb'))
        else:
            print("Overall model was not found")
        if os.path.isfile('models/modelsCols.json'):
            with open('models/modelsCols.json', 'r') as f:
                modelsColsDict = json.load(f)
        else:
            print("modelCols.json not found, reloading all models. ")
            modelsDict, modelsColsDict = getArticleModels(priceData, dailyData, stocks)
    
    articles = ripArticles(stocks)
    newDailyData = {}
    for stock in modelsDict.keys():
        if stock == 'overall':  # TODO
            continue
        newDailyData[stock] = copy.deepcopy(dailyData[stock])
        # Get stocks that have models
        model = modelsDict[stock]
        model_probability = []
        model_probability_dates = []
        
        articleDict = articles[stocks[stock]][0]
        model_adata = pd.DataFrame(columns=modelsColsDict[stock]).fillna(0)
        
        for articleDate in articleDict.keys():
            # Get articles for those stocks
            # Use articles that occur after the earliest price datapoint
            
            dt = datetime.strptime(articleDate, '%Y-%m-%d')
            dt = date(dt.year, dt.month, dt.day)
            day3 = (dt + timedelta(days=3))
            day3 = date(day3.year, day3.month, day3.day)
            day5 = (dt + timedelta(days=5))
            day5 = date(day5.year, day5.month, day5.day)
            days = [dt, day3, day5]
            for article in articleDict[articleDate]:
                temparticle = {}
                for i in article:
                    temparticle[i] = [article[i]]
                for day in days:
                    if day.strftime("%Y-%m-%d") in newDailyData[stock].index.values:
                        day_df = pd.DataFrame.from_dict(temparticle, orient='columns').rename(index={0:day.strftime("%Y-%m-%d")})
                    
                        if day.strftime("%Y-%m-%d") in model_adata.index.values:
                            model_adata = model_adata.add(day_df, fill_value=0.0)
                        else:
                            model_adata = model_adata.append(day_df, ignore_index=False, sort=False, verify_integrity=True).fillna(0)
                            
                            
                            
            # Convert adata to modelColsDict DataFrame columns
            model_adata = model_adata[modelsColsDict[stock]].fillna(0) #Removes columns not in modelsColsDict[stock]

        #Remove rows that are all zero
        model_adata = model_adata[(model_adata.T != 0).any()]
        model_probability_dates = model_adata.index.values
        
        model_probability = model.predict_proba(model_adata)
    
        model_probability_df = pd.DataFrame(data=model_probability, index=model_probability_dates,
                                            columns=['Down', 'Up'])

        for d in model_probability_dates:
            if d in newDailyData[stock].index.values:
                newDailyData[stock].loc[d,'Down'] = model_probability_df.loc[d,'Down']
                newDailyData[stock].loc[d, 'Up'] = model_probability_df.loc[d, 'Up']
        newDailyData[stock] = newDailyData[stock].fillna(0)
        if cutoff:
            firstArticleDate = sorted(articleDict.keys(), key=lambda kv: kv)[0]
            # print(newDailyData[stock])
            try:
                articleIndex = newDailyData[stock].index.get_loc(firstArticleDate)
                # print(articleIndex)
                newIndexes = newDailyData[stock].index.values[articleIndex]
                # print(newIndexes)
                newDailyData[stock] = newDailyData[stock].loc[newIndexes:]
            except:
                pass
        if cutoff:
            print("Cutting off dailyData")
    return newDailyData


def getArticleModels(priceData, dailyData, stocks):
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
    y = []
    countedAllWords = defaultdict(int)
    dataDict = defaultdict(dict)
    
    for result in results:
        if result is None:
            continue
        dataDict[result['name']] = result
        adata = adata.append(result['adata'], ignore_index=False, sort=False)
        
        y.extend(result['y'])
        for word in result['countedAllWords']:
            countedAllWords[word] += result['countedAllWords'][word]
    
    print("Classifying total words")
    
    # setup X and y
    adata = adata.fillna(0)
    y = np.array(y)
    ovAC = []
    print("Skipping overall accuracy measurments for testing ")
    if True:
        #print(len(countedAllWords))
        #print(adata.shape)
        
        # SVM with all words
        accuracies = articleSVM(adata, y, 'Overall Accuracies')
        #print("%s:  %f  %f" % (accuracies['name'], accuracies['train'], accuracies['test']))
    
    # Most common reoccuring words
    sorted_keys_by_values = sorted(countedAllWords.keys(), key=lambda kv: countedAllWords[kv], reverse=True)
    cutIndex = len(sorted_keys_by_values) // 4
    chopped_keys = sorted_keys_by_values[:cutIndex]
    
    chopped_adata = adata[chopped_keys]
    # SVM with most common words
    chopped_accuracies = articleSVM(chopped_adata, y, 'Chopped Overall Accuracies')
    #print("%s:  %f  %f" % (chopped_accuracies['name'], chopped_accuracies['train'], chopped_accuracies['test']))
    
    reducedDimResults = dimReducer(adata, y, multiThread=True)
    reducedDimResults.append(chopped_accuracies)
    reducedDimResults.append(accuracies)
    reducedDimResults = sorted(reducedDimResults, key=lambda kv: kv['test'], reverse=True)
    
    for reducedResult in reducedDimResults:
        print("%s: %f  %f" % (reducedResult['name'], reducedResult['train'], reducedResult['test']))
    if 'adata_new' in reducedDimResults[0].keys():
        total_best_fit_adata = reducedDimResults[0]['adata_new']
    else:
        total_best_fit_adata = reducedDimResults[0]['adata']
    total_best_fit_cls = reducedDimResults[0]['cls']
    total_best_fit_sv = reducedDimResults[0]['sv']
    
    print("Using None for rs in articlSVM!!!!!!")
    
    modelsDict = {}
    modelsColsDict = {}
    for stock in dataDict:
        modelsDict[stock] = dataDict[stock]['sv']
        pickle.dump(dataDict[stock]['sv'], open('models/' + stock + '.pkl', 'wb'))
        if 'adata_new' in dataDict[stock].keys():
            modelsColsDict[stock] = dataDict[stock]['adata_new'].columns.values.tolist()
        else:
            modelsColsDict[stock] = dataDict[stock]['adata'].columns.values.tolist()
    modelsDict['overall'] = total_best_fit_sv
    modelsColsDict['overall'] = total_best_fit_adata.columns.values.tolist()
    with open('models/modelsCols.json', 'w+') as f:
        json.dump(modelsColsDict, f, separators=(',', ':'))
    pickle.dump(total_best_fit_sv, open('models/overall.pkl', 'wb'))
    
    return modelsDict, modelsColsDict


def dimReducer(adata, y, multiThread=False):
    """
    Uses feature selection to reduce the number of dimensions  used.
    Multi-threaded
    :param adata:
    :param y:
    :param multiThread: Prevent child threads from spawning child threads in parseargs
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
        results = []
        for i in fargs:
            results.append(dimReducerChild(i))
    return results


def dimReducerChild(fargs):
    adata = fargs[0]
    adata = adata.fillna(0)
    y = fargs[1]
    select = fargs[2]
    classm = fargs[3]
    additionalParams = {'k_best': 200, 'percentile': 10, 'fpr': 0.005, 'fdr': 0.005, 'fwe': 0.005}
    selector = GenericUnivariateSelect(score_func=eval(classm), mode=select, param=additionalParams[select])
    adata_new = selector.fit_transform(adata, y)
    cols = selector.get_support(True)
    adata_cols = list(adata.columns.values)
    adata_new_cols = []
    for i in cols:
        adata_new_cols.append(adata_cols[i])
    adata_new = pd.DataFrame(data=adata_new, index=adata.index, columns=adata_new_cols)
    accuracies = articleSVM(adata_new, y, select + " " + classm)
    accuracies['adata_new'] = adata_new
    return accuracies


def ripArticles(stocks):
    """
    Go to NYTimes, search for company name, summarize articles and place in dict
    with key being article publish date. Returns dict of summarized articles.
    """
    
    # Get new article links
    if refreshArticles:
        articleLinks = getArticleLinks(list(stocks.values()))
        with open('summarizedArticles/articleLinks.json', 'w+') as f:
            json.dump(articleLinks, f, separators=(',', ':'))
        print("Saved article links")
    else:
        if os.path.isfile('summarizedArticles/articleLinks.json'):
            with open('summarizedArticles/articleLinks.json', 'r') as f:
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
    refArt = fargs[2]  # Clear out saved articles and summarize from scratch
    upArt = fargs[3]  # Add new articles to saved
    
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
            with open('summarizedArticles/' + stockName + '.json', 'r') as f:
                # [ {datetime:[summarized article]}, [included urls] ]
                # [ dict of lists, list ]
                data = json.load(f)
            if not upArt:
                # If not updating the articles
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
                    # Has UTC offset
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
    
    options = Options()
    options.add_argument("--profile-directory=Default")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/69.0.3497.100 Safari/537.36")
    
    driver = webdriver.Chrome(os.getcwd() + '/chromedriver.exe', options=options)
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
                if button.text == "SHOW MORE":
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
        
        """//*[@id="site-content"]/div/div/div[2]/div[2]/ol/li[143]/div/div/a"""
        
        # Get the page html source for beautiful soup parsing
        r = driver.page_source
        soup = BeautifulSoup(r, 'html.parser')
        reg = re.compile('.*css-.*')
        f = soup.find_all('li', attrs={'class': reg})
        links = []
        # Get links from search
        for i in f:
            try:
                section = i.find('p', attrs={'class': 'css-myxawk'}).text.lower()
                if section in ['technology', 'business', 'climate', 'energy & environment']:
                    links.append(quote_page + i.find('a').get('href'))
            except:
                pass
        linkDict[name] = links
        print("Number of links: %d" % len(links))
    
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
    
    # All words with appearance count
    countedAllWords = defaultdict(int)
    for articleDate in articleDict:
        for article in articleDict[articleDate]:
            for kw in article.keys():
                countedAllWords[kw] += article[kw]
    
    # All words
    allwords = list(countedAllWords.keys())
    
    # Matrix. Rows=days, Cols=words mentioned
    adata = pd.DataFrame(columns=allwords)
    
    # -1,0,1 representing changes in price
    y = []
    
    for articlesDate in articleDict:
        dt = datetime.strptime(articlesDate, '%Y-%m-%d')
        
        day0 = dt.strftime("%Y-%m-%d")
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
                if diff >= 0:
                    dayDiff.append(1)
                elif diff < 0:
                    dayDiff.append(-1)
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
        # TODO: Change adata such that it's always a DataFrame
        tempToDict = {}
        for i in articles:
            tempToDict[i] = [articles[i]]
        for day in range(0, len(dayDiff)):
            arow = pd.DataFrame.from_dict(data=tempToDict, orient='columns')
            #  arow = []
            #  for word in allwords:
            #      if word in articles.keys():
            #          arow.append(articles[word])
            #      else:
            #          arow.append(0)
            adata = adata.append(arow, ignore_index=False, sort=False).fillna(0)
            y.append(dayDiff[day])
    
    y = np.array(y)
    
    # Return if all outputs are the same
    if all(y[0] == ytemp for ytemp in y):
        print("%s All outputs the same" % stockName)
        return 0, 0, 0
    if adata.shape[1] > 700:
        accuracies = dimReducer(adata, y)
        accuracies = sorted(accuracies, key=lambda kv: kv['test'], reverse=True)[0]
        print('%s, %dx%d --> %dx%d' % (
            stockName, adata.shape[0], adata.shape[1], accuracies['adata_new'].shape[0],
            accuracies['adata_new'].shape[1]))
        if 'name' not in accuracies:
            raise Exception("Error occured when trying to reduce dimensionality. Figure something out.")
        accuracies['name'] = stockName + " reduced with " + accuracies['name']
    else:
        accuracies = articleSVM(adata, y, stockName)
    
    if accuracies['test'] < minAccuracy:
        if adata.shape[1] > 700:
            accuracies = articleSVM(adata, y, stockName)
            if accuracies['test'] < minAccuracy:
                # If accuracy is less than minimum
                print("%s had less than acceptable accuracy. %f " % (stockName, accuracies['test']))
                return None
        else:
            # If accuracy is less than minimum
            print("%s had less than acceptable accuracy. %f " % (stockName, accuracies['test']))
            return None
    print("%s:  %f  %f" % (accuracies['name'], accuracies['train'], accuracies['test']))
    
    ret = {'adata': adata, 'y': y, 'countedAllWords': countedAllWords, 'name': stockName,
           'accuracy': accuracies['test'], 'cls': accuracies['cls'], 'sv': accuracies['sv']}
    
    if 'adata_new' in accuracies:
        # Add the reduced adata to return
        # if type(accuracies['adata_new']) == np.ndarray:
        ret['adata_new'] = accuracies['adata_new']
    #   else:
    #      ret['adata_new'] = accuracies['adata_new']
    
    return ret


def articleSVM(X, y, name):
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
    rs = None  # Used to prevent infinite loop
    while all(y_train[0] == ytemp for ytemp in y_train):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rs)
        # rs += 1
    
    sv = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, \
             probability=True, cache_size=200, class_weight=None, max_iter=-1, \
             decision_function_shape='ovr', random_state=None, tol=0.00001)
    
    cls = sv.fit(X_train, y_train)
    
    accuracy_train = accuracy_score(y_train, cls.predict(X_train))
    
    accuracy_test = accuracy_score(y_test, cls.predict(X_test))
    
    accuracies = {'name': name, 'train': accuracy_train * 100, 'test': accuracy_test * 100, 'sv': sv, 'cls': cls}
    return accuracies
