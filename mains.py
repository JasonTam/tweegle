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
from collections import defaultdict
import itertools

import preprocess
import features

if __name__ == "__main__":

    import sys

    # Load data into Python Dict
    filename = sys.argv[1] if len(sys.argv) > 1 else 'data/data.json'

    with open(filename, 'r') as f:
        tweets = json.load(f)

        # Preprocess data from json into data structs
        docs_terms, docs_t_collection = preprocess.setup_doc_collection(tweets)

        # Getting location info
        targets = preprocess.get_target_map(tweets)
        loc_to_id = defaultdict(set)
        for k, v in targets.iteritems():
            loc_to_id[v].add(k)
        all_locations = set(loc_to_id.keys())

        # Combine tweets from same location
        mega_docs = {loc:
                     list(itertools.chain(
                         *[docs_terms[id].tokens for id in loc_to_id[loc]]
                     ))
                     for loc in all_locations
                    }

        tfidf = TfidfVectorizer(stop_words='english')

        # Get feature vector for a given location as the mean of tfidf
        tfs = tfidf.fit_transform(docs_t_collection.tokens)
        loc_feats = {}
        
        for loc in all_locations:
            loc_feats[loc] = tfidf.transform(mega_docs[loc]).mean(0)