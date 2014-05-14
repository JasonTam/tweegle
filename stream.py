import json
import tweepy
from datetime import datetime
import time


def valid_tweet(tweet):
    check_coord = 'coordinates' in tweet
    check_place = 'place' in tweet
    if check_place:
        check_place = tweet['place'] is not None
    check_reply = not tweet['in_reply_to_user_id']
    check_rt = not tweet['retweeted']

    return check_coord and check_place and check_reply and check_rt


def convert_tweet(tweet):
    info = {k: tweet[k] for k in ['id', 'coordinates', 'text', 'place']}
    info['user'] = {k: tweet['user'][k] for k in ['location', 'name', 'screen_name', 'time_zone', 'lang']}
    info['text'] = info['text'].encode('utf-8')
    info['time'] = time.mktime(datetime.strptime(tweet['created_at'],
                                                 '%a %b %d %H:%M:%S +0000 %Y').timetuple())
    return info


class SaveTweets(tweepy.StreamListener):
    def __init__(self, filename, limit):
        self.filename = filename
        self.limit = limit
        self.tweets = []
        self.debug = True
        self.batch_size = 100
        self.num_saved = 0

    def on_status(self, tweet):
        pass

    def on_error(self, status_code):
        print repr(status_code)
        return False

    def on_data(self, data):

        if data:
            tweet = json.loads(data)

            # Ignore replies and retweets
            if valid_tweet(tweet):
                info = convert_tweet(tweet)

                self.tweets.append(info)
                n_tweets = len(self.tweets)
                if self.debug:
                    dbg_str = '\r# Tweets: ' + str(n_tweets)
                    sys.stdout.write(dbg_str)
                    sys.stdout.flush()

                if n_tweets < self.limit:
                    if (n_tweets % self.batch_size) == 0:
                        with open(self.filename, 'w') as f:
                            json.dump(self.tweets, f)

                    return True
                else:
                    return False


if __name__ == "__main__":
    import secret
    import sys

    auth = tweepy.OAuthHandler(secret.api_key, secret.api_secret)
    auth.set_access_token(secret.access_token_key, secret.access_token_secret)

    filename = sys.argv[1] if len(sys.argv) > 1 else 'data/data.json'
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    listener = SaveTweets(filename=filename, limit=limit)
    streamer = tweepy.Stream(auth=auth, listener=listener)

    locations = [-180, -90, 180, 90]
    streamer.filter(locations=locations)

# PATCH?
import httplib

def patch_http_response_read(func):
    def inner(*args):
        try:
            return func(*args)
        except httplib.IncompleteRead, e:
            return e.partial

    return inner
httplib.HTTPResponse.read = patch_http_response_read(httplib.HTTPResponse.read)