# -*- coding: utf-8 -*-

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from nltk.stem import SnowballStemmer

### PREPROCESSING THE TWEET FOR ANALYSIS ###
def porter_process(tweet, lowercase = True, stem = True, stopwords = True, gram = 2):
    
    if lowercase:
        tweet = tweet.lower()
    words = word_tokenize(tweet)
    words = [w for w in words if len(w) > 2]
    
    
    if gram > 1:
        w = []
        for i in range(len(words) - gram + 1):
            w += [' '.join(words[i:i + gram])]
        return w
    
    
    if stopwords:
        stoppedW = stopwords.words('english')
        words = [word for word in words if word not in stoppedW]
    
    
    if stem:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]
    return words

### SNOWBALL PROCESSING, SIMPLE AND QUICKER ###
def snowball_process(tweet):
    tweet = tweet.translate(string.maketrans('',''),'!"#$%&()*+, -./:;<=>?@[\]^_`{|}~')
    tweet = [word for word in tweet.split() if word.lower() not in stopwords.words('english')]
    words = ""
    for i in tweet:
            stemmer = SnowballStemmer("english")
            words += (stemmer.stem(i))+" "
    return words
