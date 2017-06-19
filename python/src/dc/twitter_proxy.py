import logging
import sys
import json
import os
import urllib.request

import datetime
from SolrClient import SolrClient
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
from geopy.geocoders import Nominatim

IGNORE_RETWEETS=True
LANGUAGES_ACCETED=["en"]
LOG_DIR=os.getcwd()+"/logs"
SOLR_SERVER="http://localhost:8983/solr"
SOLR_CORE="chase"
TWITTER_TIME_PATTERN="%a %b %d %H:%M:%S %z %Y"
SOLR_TIME_PATTERN="%Y-%m-%dT%H:%M:%SZ" #YYYY-MM-DDThh:mm:ssZ
LOCATION_COORDINATES={} #cache to look up location geocodes
geolocator = Nominatim()
logger = logging.getLogger(__name__)
logging.basicConfig(filename=LOG_DIR+'/twitter_stream.log', level=logging.INFO, filemode='w')

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



class TwitterStream(StreamListener):
    __solr = None
    __core=None
    __count=0
    __count_retweet=0

    def __init__(self):
        super().__init__()
        self.__solr=SolrClient(SOLR_SERVER)
        self.__core=SOLR_CORE

    def ignoreRetweet(self, status_text):
        if "rt @" in status_text.lower() and IGNORE_RETWEETS:
            self.__count_retweet+=1
            return True
        return False

    def on_data(self, data):
        self.__count+=1
        jdata = None
        try:
            jdata = json.loads(data)
            if jdata is not None and "id" in jdata.keys() and not self.ignoreRetweet(jdata["text"]):
                #created_at_time
                str_created_at= jdata["created_at"]
                time=datetime.datetime.strptime(str_created_at, TWITTER_TIME_PATTERN)
                str_solr_time=time.utcnow().strftime(SOLR_TIME_PATTERN)

                #entities hashtags
                hashtags=jdata["entities"]["hashtags"]
                hashtag_list=[]
                for hashtag in hashtags:
                    hashtag_list.append(hashtag["text"])

                #entities urls
                urls=jdata["entities"]["urls"]
                url_list=[]
                for url in urls:
                    url_list.append(url["expanded_url"])

                #entities symbols
                symbols=jdata["entities"]["symbols"]
                symbols_list=[]
                for symbol in symbols:
                    symbols_list.append(symbol["text"])

                #entities user_mentions
                user_mentions=jdata["entities"]["user_mentions"]
                user_mention_list=[]
                for um in user_mentions:
                    user_mention_list.append(um["id"])

                #user_location
                str_user_loc=jdata["user"]["location"]
                if str_user_loc in LOCATION_COORDINATES.keys():
                    geocode_obj = LOCATION_COORDINATES[str_user_loc]
                else:
                    geocode_obj=None #currently the api for getting geo codes seems to be unstable
                    # geocode_obj = geolocator.geocode(str_user_loc)
                    # LOCATION_COORDINATES[str_user_loc]=geocode_obj
                geocode_coordinates=[]
                if geocode_obj is not None:
                    geocode_coordinates.append(geocode_obj.latitude)
                    geocode_coordinates.append(geocode_obj.longitude)

                #quoted status id if exists
                if "quoted_status_id" in jdata:
                    quoted_status_id=jdata["quoted_status_id"]
                else:
                    quoted_status_id=None

                #place exists
                place=jdata["place"]
                if place is not None:
                    place_full_name=place["full_name"]
                    place_coordinates=place['bounding_box']['coordinates'][0][0]
                else:
                    place_full_name=None
                    place_coordinates=None

                docs = [{'id':jdata["id"],
                       'created_at':str_solr_time,
                       'geo':jdata["geo"],
                       'coordinates':jdata["coordinates"],
                       'favorite_count':jdata["favorite_count"],
                       'in_reply_to_screen_name':jdata["in_reply_to_screen_name"],
                       'in_reply_to_status_id':jdata["in_reply_to_status_id"],
                       'in_reply_to_user_id':jdata["in_reply_to_user_id"],
                       'lang':jdata["lang"],
                       'place_full_name':place_full_name,
                       'place_coordinates':place_coordinates,
                       'retweet_count':jdata["retweet_count"],
                       'retweeted':jdata["retweeted"],
                       'quoted_status_id':quoted_status_id,
                       'status_text':jdata["text"],
                       'entities_hashtag':hashtag_list,
                       'entities_symbol':symbols_list,
                       'entities_url':url_list,
                       'entities_user_mention':user_mention_list,
                       'user_id':jdata["user"]["id"],
                       'user_screen_name':jdata["user"]["screen_name"],
                       'user_statuses_count':jdata["user"]["statuses_count"],
                       'user_friends_count':jdata["user"]["friends_count"],
                       'user_followers_count':jdata["user"]["followers_count"],
                       'user_location':str_user_loc,
                       'user_location_coordinates':geocode_coordinates}]
                self.__solr.index(self.__core,docs)

                if self.__count%500==0:
                    code=urllib.request.\
                        urlopen("http://localhost:8983/solr/{}/update?commit=true".
                                format(self.__core)).read()
                    now=datetime.datetime.now()
                    print("{} processed: {}, where {} are retweets and ignored".
                          format(now, self.__count, self.__count_retweet))
                    logger.info("{} processed: {}, where {} are retweets and ignored".
                                format(now,self.__count, self.__count_retweet))
        except Exception as exc:
            print("Error encountered for {}, error:{} (see log file for details)".format(self.__count, exc))
            if jdata is not None and "id" in jdata.keys():
                tweet_id=jdata["id"]
            else:
                tweet_id="[failed to parse]"
            logger.info("Error encountered for counter={}, tweet={}, error:{} (see log file for details)".
                        format(self.__count, tweet_id, exc))
            if jdata is not None:
                file = LOG_DIR+"/"+str(tweet_id)+".txt"
                logger.info("\t input data json written to {}".format(file))
                with open(file, 'w') as outfile:
                    json.dump(jdata, outfile)
            pass
        return(True)

    def on_error(self, status):
        print(status)

    def on_status(self, status):
        print(status.text)

oauth=read_oauth(sys.argv[1])
print(sys.argv[1])
sc=read_search_criteria(sys.argv[2])
print(sys.argv[2])
auth = OAuthHandler(oauth["C_KEY"], oauth["C_SECRET"])
auth.set_access_token(oauth["A_TOKEN"], oauth["A_SECRET"])

twitterStream = Stream(auth, TwitterStream())
twitterStream.filter(track=[sc["KEYWORDS"]], languages=LANGUAGES_ACCETED)

