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

            info = {}
            info['user'] = tweet['user']
            info['coordinates'] = tweet['coordinates']
            info['text'] = tweet['text']
            info['place'] = tweet['place']

            self.tweets.append(info)

            if len(self.tweets) < self.limit:

                with open(self.filename, 'w') as f:
                    json.dump(self.tweets, f)

                return True
            else:
                return False


if __name__ == "__main__":

    import secret

    auth = tweepy.OAuthHandler(secret.api_key, secret.api_secret)
    auth.set_access_token(secret.access_token_key, secret.access_token_secret)

    listener = StreamListener(filename='data/data.json', limit=50)
    streamer = tweepy.Stream(auth=auth, listener=listener)

    locations = [-180, -90, 180, 90]
    streamer.filter(locations=locations)