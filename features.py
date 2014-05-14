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
from sklearn.metrics.pairwise import cosine_similarity
import sys

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
                            sublinear_tf=True,
                            smooth_idf=True,
                            ngram_range=(1, 2))
    tfs = tfidf.fit_transform(doc_list)
    return tfidf


def cosine_sim_all(test_raw, test_doc_tfidf_feat, loc_feats, debug=False):
    sim = {}
    sim_tweet = {}
    sim_pred = {}
    # Cosine Similarity for every possibility
    for ii, t_id in enumerate(test_raw.keys()):
        if debug:
            dbg_str = '\rCosine Sim: ' + str(t_id) + '[' + str(ii + 1) + '/' + str(len(test_raw)) + ']'
            sys.stdout.write(dbg_str)
            sys.stdout.flush()

        sim_pred[t_id], sim_tweet[t_id] = cosine_sim_single(
            test_doc_tfidf_feat[t_id], loc_feats
        )
    return sim_pred, sim_tweet


def cosine_sim_single(doc_tfidf_feat, loc_feats):
    all_locations = loc_feats.keys()
    sim = {}
    for jj, loc in enumerate(all_locations):
        sim[loc] = cosine_similarity(doc_tfidf_feat, loc_feats[loc])

    sims = sim.values()
    pred = max(sim, key=sim.get)
    return pred, sims







