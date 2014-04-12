
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

import preprocess
import features



# Load data into Python Dict
dir_data = './data'
filename = 'test.json'
filepath = os.path.join(dir_data, filename)
with open(filepath, 'r') as f:
    tweets = json.load(f)


docs_terms, docs_t_collection = preprocess.setup_doc_collection(tweets)
targets = preprocess.get_target_map(tweets)






