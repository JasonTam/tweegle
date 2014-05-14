tweegle
=======

Automatic geo-location for tweets using natural language processing

requirements
------------
Python 2.7 with NLTK, scikit-learn, numpy, and tweepy.

setup
-----
Create a `secret.py` file with twitter app tokens:
```
api_key = '<api key>'
api_secret = '<api secret>'

access_token_key = '<access token key>'
access_token_secret = '<access token secret>'
```

run
---
To retrieve a corpus of tweets, run:
```
$ stream.py <filename> <number of tweets>
```

To train the classifier based on the corpus, run:
```
$ mains.py <training corpus> <test corpus>
```

To see it classify tweets in real time, run:
```
$ demo.py
```
