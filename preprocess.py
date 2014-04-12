

import json

import numpy as np
import string
import nltk
from nltk.corpus import PlaintextCorpusReader
from nltk import text
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize.punkt import PunktWordTokenizer

import os
import time


# Load data into Python Dict
dir_data = './data'
filename = 'test.json'
filepath = os.path.join(dir_data, filename)
with open(filepath, 'r') as f:
    tweets = json.load(f)

def prune_unicode(text):
    return ''.join([i if ord(i) < 128 else '' for i in text])

def setup_doc_collection(tweets,
                         toknzr=PunktWordTokenizer(),
                         stemmer=nltk.PorterStemmer()):
    start = time.time()
    if stemmer:
        get_terms = lambda raw_terms: \
            [stemmer.stem(w.lower()) for w in raw_terms]
    else:
        get_terms = lambda raw_terms: \
            [w.lower() for w in raw_terms]

    # Setup Text docs to contain frequent terms
    docs_terms = {}
    for tweet in tweets:
        raw = tweet['text']
        raw = prune_unicode(raw)
        raw_tokens = toknzr.tokenize(raw)
        terms = get_terms(raw_tokens)
        docs_terms[tweet['id']] = text.Text(terms)

    docs_t_collection = text.TextCollection(docs_terms.values())
    print 'Time to prune corpus:' + str(time.time() - start)
    return docs_terms, docs_t_collection


def get_target_map(tweets):
    tweet_classes = {
        tweet['id']: tweet['place']['country_code']
        for tweet in tweets
    }
    return tweet_classes



