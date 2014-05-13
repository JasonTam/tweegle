import json

import numpy as np
import string
import nltk
from nltk.corpus import PlaintextCorpusReader
from nltk import text
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
from nltk.tokenize.punkt import PunktWordTokenizer
from sklearn import preprocessing

import os
import time
from collections import defaultdict
import itertools

import preprocess
import features
import classification
import validate

debug = True


# 455147014511403008 is an example of a tweet we should get right
# (NYC)
# test_tweets
def get_tweet_with_id(tweet_list, id):
    """
    Just a function used for debugging
    """
    q = [tweet_list[ii] for ii in range(len(tweet_list))
         if tweet_list[ii]['id'] == id]
    return q[0]


if __name__ == "__main__":

    # Load train data into Python Dict
    import sys

    train = sys.argv[1] if len(sys.argv) > 1 else 'data/train.json'
    with open(train, 'r') as f:
        tweets = json.load(f)

# [PREPROCESS]---------------------------------------
    # Preprocess data from json into data structs
    if debug:
        print '\nPreprocessing: Setting up Train Doc Collection'
    train_raw = preprocess.setup_doc_collection(tweets)

    # Getting location info
    targets = preprocess.get_target_map(tweets)
    loc_to_id = defaultdict(set)
    for k, v in targets.iteritems():
        loc_to_id[v].add(k)
    all_locations = sorted(loc_to_id.keys())

    classifier = classification.Classifier()
    classifier.le.fit(all_locations)

# [MEGADOCS]---------------------------------------
    # Combine tweets from same location
    mega_raws = {
        loc: [train_raw[t_id] for t_id in loc_to_id[loc]]
        for loc in all_locations
    }

# [TFIDF TRAIN MEGADOCS]---------------------------------------
    tfidf = features.fit_tfidf(train_raw.values())

    loc_feats = {}
    for ii, loc in enumerate(all_locations):
        if debug:
            dbg_str = '\rTFIDF Train MegaDocs: ' + loc + '[' + str(ii + 1) + '/' + str(len(all_locations)) + ']'
            sys.stdout.write(dbg_str)
            sys.stdout.flush()
        loc_feats[loc] = tfidf.transform(mega_raws[loc]).mean(0)


# [TFIDF ON TRAINING TWEETS]---------------------------------------
    train_doc_tfidf_feat = {}
    for ii, t_id in enumerate(train_raw.keys()):
        if debug:
            dbg_str = '\rTFIDF Train Tweets: ' + str(t_id) + '[' + str(ii + 1) + '/' + str(len(train_raw)) + ']'
            sys.stdout.write(dbg_str)
            sys.stdout.flush()
        train_doc_tfidf_feat[t_id] = tfidf.transform([train_raw[t_id]])

        classifier.add_training_data(
            train_doc_tfidf_feat[t_id],
            classifier.le.transform([targets[t_id]])[0])
    classifier.fit()


## --------------[ TESTING ]---------------
    # Load test data into Python Dict
    test = sys.argv[2] if len(sys.argv) > 2 else 'data/test.json'
    with open(test, 'r') as f:
        test_tweets = json.load(f)

    if debug:
        print '\nPreprocessing: Setting up Test Doc Collection'
    test_raw = preprocess.setup_doc_collection(test_tweets)

    test_doc_tfidf_feat = {}
    for ii, t_id in enumerate(test_raw.keys()):
        if debug:
            dbg_str = '\rTFIDF: ' + str(t_id) + '[' + str(ii + 1) + '/' + str(len(test_raw)) + ']'
            sys.stdout.write(dbg_str)
            sys.stdout.flush()
        test_doc_tfidf_feat[t_id] = tfidf.transform([test_raw[t_id]])

    sim_pred, sim_test = features.cosine_sim(
        test_raw, test_doc_tfidf_feat, all_locations, loc_feats, debug=debug)

    predictions = {}
    predictions_loc = {}
    for ii, t_id in enumerate(test_raw.keys()):
        if debug:
            dbg_str = '\rPredicting: ' + str(t_id) + '[' + str(ii + 1) + '/' + str(len(test_raw)) + ']'
            sys.stdout.write(dbg_str)
            sys.stdout.flush()
        predictions[t_id] = classifier.predict(test_doc_tfidf_feat[t_id].todense())

    for k, v in predictions.iteritems():
        predictions_loc[k] = classifier.le.inverse_transform(v)


print '\n'
# Validation
y_pred, y_truth = zip(*[(sim_pred[tweet['id']], tweet['place']['country_code'])
                        for tweet in test_tweets])
validator = validate.Validator(y_truth, y_pred)
print validator.acc
