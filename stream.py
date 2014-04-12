import json
import tweepy

class StreamListener(tweepy.StreamListener):

    def __init__(self, filename, limit):
        self.filename = filename
        self.limit = limit
        self.tweets = []

    def on_status(self, tweet):
        pass

    def on_error(self, status_code):
        print repr(status_code)
        return False

    def on_data(self, data):

        if data:
            tweet = json.loads(data)

            user = tweet['user']

            info = {k:tweet[k] for k in ['coordinates', 'text', 'place']}
            info['user'] = {k:tweet['user'][k] for k in ['location', 'name', 'screen_name', 'time_zone']}

            self.tweets.append(info)

            if len(self.tweets) < self.limit:

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
    limit = sys.argv[2] if len(sys.argv) > 2 else 50

    listener = StreamListener(filename=filename, limit=limit)
    streamer = tweepy.Stream(auth=auth, listener=listener)

    locations = [-180, -90, 180, 90]
    streamer.filter(locations=locations)