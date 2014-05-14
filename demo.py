import stream
import tweepy
import json
import pickle
import features


class ProcessTweets(tweepy.StreamListener):
    def __init__(self):
            # Save training
        save_xform_path = './data/tfidf.p'
        save_loc_feats_path = './data/loc_feats.p'
        self.tfidf = pickle.load(open(save_xform_path, "rb"))
        self.loc_feats = pickle.load(open(save_loc_feats_path, "rb"))

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
                print 'Text:', info['text']
                print 'Country:', info['place']['country_code']
                print 'Prediction:', self.predict(info['text'])

                raw_input('...')
                return True  # True to continue

    def predict(self, info_text):
        feat = self.tfidf.transform([info_text])
        pred, sims = features.cosine_sim_single(feat, self.loc_feats)
        return pred



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