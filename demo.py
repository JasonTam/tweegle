import stream
import tweepy
import json

class ProcessTweets(tweepy.StreamListener):
    def on_status(self, tweet):
        pass

    def on_error(self, status_code):
        print repr(status_code)
        return False

    def on_data(self, data):

        if data:
            tweet = json.loads(data)

            # Ignore replies and retweets
            if stream.valid_tweet(tweet):
                info = stream.convert_tweet(tweet)

                # Process tweet here
                print info

                return True # True to contineu


if __name__ == "__main__":
    import secret
    import sys

    auth = tweepy.OAuthHandler(secret.api_key, secret.api_secret)
    auth.set_access_token(secret.access_token_key, secret.access_token_secret)

    filename = sys.argv[1] if len(sys.argv) > 1 else 'data/data.json'
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    listener = ProcessTweets()
    streamer = tweepy.Stream(auth=auth, listener=listener)

    locations = [-180, -90, 180, 90]
    streamer.filter(locations=locations)