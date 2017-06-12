from tweepy import Stream
from tweepy import OAuthHandler
from dc import twitter_stream as ts
import sys


def read_oauth(file):
    vars = {}
    with open(file) as auth_file:
        for line in auth_file:
            name, var = line.partition("=")[::2]
            vars[name.strip()] = str(var).strip()
    return vars

def read_search_criteria(file):
    vars = {}
    with open(file) as auth_file:
        for line in auth_file:
            name, var = line.partition("=")[::2]
            vars[name.strip()] = str(var).strip()
    return vars


oauth=read_oauth(sys.argv[1])
sc=read_search_criteria(sys.argv[2])
auth = OAuthHandler(oauth["C_KEY"], oauth["C_SECRET"])
auth.set_access_token(oauth["A_TOKEN"], oauth["A_SECRET"])

twitterStream = Stream(auth, ts.TwitterStream())
twitterStream.filter(track=[sc["KEYWORDS"]])

