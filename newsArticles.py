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

def getArticles(priceData,stocks):
    """
    main func for articles
    """
    print("Ripping and summariing articles")
    alldata = ripArticles(stocks)  # Get articles and summarize and keywords
    wordWeights = defaultdict(int)

    print("Classifying keywords")

    results = []
    fargs = []
    pool = Pool()

    for stockName in stocks.keys():
        if alldata[stocks[stockName]][0]:
            fargs.append([alldata[stocks[stockName]][0], priceData[stockName], stockName])

    results = pool.map(parseArticles, fargs)
    pool.close()
    pool.join()

    adata = pd.DataFrame()
    y = []
    allwords = set()

    for result in results:
        tadata, ty, tallWords = result
        try:
            if tadata == 0:
                continue
        except:
            pass
        adata = adata.append(tadata, ignore_index=True)
        y.extend(ty)
        for word in tallWords:
            allwords.add(word)
    print("Classifying total words")
    adata = adata.fillna(0)
    y = np.array(y)

    y_train = [0, 0]
    rs = 0  # Used to prevent infinite loop
    while all(y_train[0] == ytemp for ytemp in y_train):
        X_train, X_test, y_train, y_test = train_test_split(adata, y, test_size=0.2, random_state=rs)
        rs += 1

    sv = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, \
             probability=True, cache_size=200, class_weight=None, max_iter=-1, \
             decision_function_shape='ovr', random_state=0, tol=0.001)

    cls = sv.fit(X_train, y_train)

    accuracy_train = accuracy_score(y_train, cls.predict(X_train))

    accuracy_test = accuracy_score(y_test, cls.predict(X_test))

    accuracies = ['Overall Accuracies', accuracy_train * 100, accuracy_test * 100]
    print(accuracies)
    print(sv.score(X_test, y_test))
    # TODO Shared words classification

    # TODO return_estimator

    1 / 0

    return



def parseArticles(fargs):
    articleDict = fargs[0]
    dailyData = fargs[1]
    stockName = fargs[2]
    # Dict of article keywords and occurances sorted by publish date of  article
    # Daily price data

    # Set of all keywords found across all articles of a company
    allwords = set()
    for articleDate in articleDict:
        for article in articleDict[articleDate]:
            for kw in article.keys():
                allwords.add(kw)

    allwords = list(allwords)
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

    y_train = [0, 0]
    rs = 0  # Used to prevent infinite loops
    while all(y_train[0] == ytemp for ytemp in y_train):
        X_train, X_test, y_train, y_test = train_test_split(adata, y, test_size=0.2, random_state=rs)
        rs += 1

    sv = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, \
             probability=True, cache_size=200, class_weight=None, max_iter=-1, \
             decision_function_shape='ovr', random_state=0, tol=0.001)

    cls = sv.fit(X_train, y_train)

    accuracy_train = accuracy_score(y_train, cls.predict(X_train))

    accuracy_test = accuracy_score(y_test, cls.predict(X_test))

    accuracies = [stockName, accuracy_train * 100, accuracy_test * 100]
    if accuracy_test * 100 < 30:
        print("%s had less than acceptable accuracy. %f " % (stockName, accuracy_test * 100))
        return 0, 0, 0
    print(accuracies)

    return [adata, y, allwords]





def getArticleLinks(names):
    print("Getting article links")
    driver = webdriver.Chrome()
    qpage = 'https://www.nytimes.com/search?query='
    quote_page = 'https://www.nytimes.com/'
    linkDict = {}
    for name in names:
        print(name)
        links = []
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
                print("Can't find path")
                time.sleep(1)
                pass

        if not button:
            # If button was not found, go to next stock
            continue

        # Scroll the page down
        ActionChains(driver).key_down(Keys.PAGE_DOWN)
        time.sleep(0.5)
        for i in range(20):
            clicked = False
            for lcv in range(7):
                try:
                    button.click()
                    time.sleep(0.5)
                    clicked = True
                    break
                except:
                    print("Can't find button")
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

    # Gracefully close driver
    driver.close()

    return linkDict


def ripArticlesChild(fargs):
    """
    Search NYTimes for a company.
    Read all applicable articles.
    Summarize all articles.

    """
    stockName = fargs[0]
    links = fargs[1]
    refreshArticles=fargs[2]

    # https://github.com/miso-belica/sumy
    # XPATH of parent container of article body=    //*[@id="story"]/section
    # XPATH of first body subsection=     //*[@id="story"]/section/div[1]
    quote_page = 'https://www.nytimes.com/'
    qpage = 'https://www.nytimes.com/search?query='

    if refreshArticles:
        data = [defaultdict(list), []]
    else:
        if os.path.isfile('summarizedArticles/' + stockName + '.json'):
            with open('summarizedArticles/' + stockName + '.json','r') as f:
                # [ {datetime:[summarized article]}, [included urls] ]
                # [ dict of lists, list ]
                data = json.load(f)
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
            except:
                articleDate = articleDate.replace("Sept", "Sep")
                dt = datetime.strptime(articleDate, "%b. %d, %Y")
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

    return stockName, data


def ripArticles(stocks):
    """
    Go to NYTimes, search for company name, summarize articles and place in dict
    with key being article publish date. Returns dict of summarized articles.
    """

    #Get new article links
    refreshArticles = True
    if refreshArticles:
        articleLinks = getArticleLinks(list(stocks.values()))
        with open('summarizedArticles/articleLinks.json', 'w+') as f:
            json.dump(articleLinks, f, separators=(',', ':'))
    else:
        if os.path.isfile('summarizedArticles/articleLinks.json'):
            with open('summarizedArticles/articleLinks.json','r') as f:
                articleLinks = json.load(f)
        else:
            articleLinks = getArticleLinks(list(stocks.values()))
            with open('summarizedArticles/articleLinks.json', 'w+') as f:
                json.dump(articleLinks, f, separators=(',', ':'))

    pool = Pool()
    alldata = {}
    fargs = []

    for stockName in stocks.values():
        fargs.append([stockName, articleLinks[stockName], refreshArticles])
    results = pool.map(ripArticlesChild, fargs)
    pool.close()
    pool.join()
    for i in results:
        alldata[i[0]] = i[1]

    return alldata
