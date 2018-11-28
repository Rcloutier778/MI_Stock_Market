"""
Simple-Summarizer
A very simple summarizier built using NLP in Python
https://jodylecompte.com
https://github.com/jodylecompte/Simple-Summarizer
"""
import argparse

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.probability import FreqDist
from heapq import nlargest
from collections import defaultdict
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.summarization import keywords

def getSummary(inputContent):
    """ Drive the process from argument to output """ 

    content = sanitize_input(inputContent)

    sentence_tokens, word_tokens = tokenize_content(content)  
    sentence_ranks = score_tokens(word_tokens, sentence_tokens)
    summary = summarize(sentence_ranks, sentence_tokens)
    return summary
    
def getKeywords(text):
    """
    Get keywords of text with the count of the number of times they appear
    """
    #TODO Add plural stripping (convert plural words to singular to help reduce number of dimensions)
    kwordsCount={}
    kwords = keywords(text).strip().split('\n')
    for word in kwords:
        kwordsCount[word]=text.count(word)
    return kwordsCount
    
    
    
    
def sanitize_input(data):
    """ 
    Currently just a whitespace remover. More thought will have to be given with how 
    to handle sanitzation and encoding in a way that most text files can be successfully
    parsed
    """
    replace = {
        ord('\f') : ' ',
        ord('\t') : ' ',
        ord('\n') : ' ',
        ord('\r') : None
    }

    return data.translate(replace)

def tokenize_content(content):
    """
    Accept the content and produce a list of tokenized sentences, 
    a list of tokenized words, and then a list of the tokenized words
    with stop words built from NLTK corpus and Python string class filtred out. 
    """
    stop_words = set(stopwords.words('english') + list(punctuation))
    words = word_tokenize(content.lower())
    
    return [
        sent_tokenize(content),
        [word for word in words if word not in stop_words]    
    ]

def score_tokens(filterd_words, sentence_tokens):
    """
    Builds a frequency map based on the filtered list of words and 
    uses this to produce a map of each sentence and its total score
    """
    word_freq = FreqDist(filterd_words)

    ranking = defaultdict(int)

    for i, sentence in enumerate(sentence_tokens):
        for word in word_tokenize(sentence.lower()):
            if word in word_freq:
                ranking[i] += word_freq[word]

    return ranking

def summarize(ranks, sentences):
    """
    Utilizes a ranking map produced by score_token to extract
    the highest ranking sentences in order after converting from
    array to string.  
    """


    indexes = nlargest(len(sentences), ranks, key=ranks.get)
    final_sentences = [sentences[j] for j in sorted(indexes)]
    return ' '.join(final_sentences) 
