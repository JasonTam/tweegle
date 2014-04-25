__author__ = 'jason'


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


def stem_tokens(tokens, stemmer=nltk.PorterStemmer()):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(raw, toknzr=PunktWordTokenizer()):
    tokens = toknzr.tokenize(raw)
    stems = stem_tokens(tokens)
    return stems


def fit_tfidf(doc_list):
    # should be run on all training docs (all locations)
    tfidf = TfidfVectorizer(tokenizer=tokenize,
                            stop_words='english',
                            lowercase=True,
                            sublinear_tf=True)
    tfs = tfidf.fit_transform(doc_list)
    return tfidf

