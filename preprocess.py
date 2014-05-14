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

import unicodedata


def fix_unicode(text):
    # return ''.join([i if ord(i) < 128 else 'u' for i in text])
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore')


def setup_doc_collection(tweets):
    docs_text = {}
    for tweet in tweets:
        # raw = fix_unicode(tweet['text'])
        raw = tweet['text']
        low = raw.lower()
        # docs_text[tweet['id']] = low.translate(None, string.punctuation)
        docs_text[tweet['id']] = low
    return docs_text


def country_code_lookup(tweets, loc_to_id):
    country_code_dict = {}
    for code in loc_to_id.keys():
        country_code_dict[code] = \
            [tweet['place']['country'] for tweet in tweets
             if tweet['id'] == list(loc_to_id[code])[0]][0]
    return country_code_dict


def get_time(tweets):
    tweet_times = {
        tweet['id']: tweet['time']
        for tweet in tweets
    }
    return tweet_times


def get_target_map(tweets):
    tweet_classes = {
        tweet['id']: tweet['place']['country_code']
        # tweet['id']: tweet['place']['name']
        for tweet in tweets
    }
    return tweet_classes


if __name__ == "__main__":
    import sys

    filename = sys.argv[1] if len(sys.argv) > 1 else 'data/data.json'
    with open(filename, 'r') as f:
        terms, collection = setup_doc_collection(json.load(f))
        print terms, collection
