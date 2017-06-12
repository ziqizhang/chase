from tweepy.streaming import StreamListener
import json


class TwitterStream(StreamListener):

    def on_data(self, data):
        print(data)
        json_data = json.loads(data)
        return(True)

    def on_error(self, status):
        print(status)
