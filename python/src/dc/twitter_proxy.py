import logging
import random
import sys
import json
import os
import traceback
import urllib.request

import datetime

import tweepy
from SolrClient import SolrClient
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener, Stream
from geopy.geocoders import Nominatim
from index import util as iu

# documentation of tweets fields https://dev.twitter.com/overview/api/tweets
# from dc import util
# from ml import util as mutil
# from ml.vectorizer import fv_chase_basic

IGNORE_RETWEETS = True
LANGUAGES_ACCETED = ["en"]
SOLR_CORE_SEARCHAPI = "chase_searchapi"
TWITTER_TIME_PATTERN = "%a %b %d %H:%M:%S %z %Y"
SOLR_TIME_PATTERN = "%Y-%m-%dT%H:%M:%SZ"  # YYYY-MM-DDThh:mm:ssZ
LOCATION_COORDINATES = {}  # cache to look up location geocodes
geolocator = Nominatim()
LOG_DIR = os.getcwd() + "/logs"
logger = logging.getLogger(__name__)
logging.basicConfig(filename=LOG_DIR + '/twitter_stream.log', level=logging.INFO, filemode='w')
# feat_vectorizer=fv_chase_basic.FeatureVectorizerChaseBasic()
SCALING_STRATEGY = -1


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


class TwitterSearch():
    __solr = None
    __core = None
    __api = None

    def __init__(self, oauth):
        super().__init__()
        self.__solr = SolrClient(iu.solr_url)
        self.__core = iu.solr_core_tweets
        self.__api = tweepy.API(oauth)

    def index(self, keywords):
        for keyword in keywords:
            count = 0
            for status in tweepy.Cursor(self.__api.search, q=keyword,
                                        tweet_mode="extended", lang="en").items(500):
                count += 1
                # created_at_time
                str_created_at = status.created_at
                str_solr_time = str_created_at.utcnow().strftime(SOLR_TIME_PATTERN)
                docs = [{'id': status.id,
                         'created_at': str_solr_time,
                         'status_text': status.full_text}
                        ]
                self.__solr.index(self.__core, docs)
            print(str(count) + "," + keyword)
        code = iu.commit(iu.solr_core_tweets)


class TwitterStream(StreamListener):
    __solr = None
    __core = None
    __count = 0
    __count_retweet = 0

    def __init__(self, ml_model_file, ml_selected_features, sysout):
        super().__init__()
        self.__solr = SolrClient(iu.solr_url)
        self.__core = iu.solr_core_tweets
        # self.__ml_model=util.load_ml_model(ml_model_file)
        # self.__selected_features = mutil.read_preselected_features(False, ml_selected_features)
        self.__sysout = sysout

    def ignoreRetweet(self, status_text):
        if "rt @" in status_text.lower() and IGNORE_RETWEETS:
            self.__count_retweet += 1
            return True
        return False

    def on_data(self, data):
        self.__count += 1
        if self.__count % 100 == 0:
            code = urllib.request. \
                code = iu.commit(iu.solr_core_tweets)
            now = datetime.datetime.now()
            print("{} processed: {}, where {} are retweets and ignored".
                  format(now, self.__count, self.__count_retweet))
            logger.info("{} processed: {}, where {} are retweets and ignored".
                        format(now, self.__count, self.__count_retweet))
        jdata = None
        try:
            jdata = json.loads(data)
            if jdata is not None and "id" in jdata.keys() and not self.ignoreRetweet(jdata["text"]):
                # created_at_time
                str_created_at = jdata["created_at"]
                time = datetime.datetime.strptime(str_created_at, TWITTER_TIME_PATTERN)
                str_solr_time = time.utcnow().strftime(SOLR_TIME_PATTERN)

                # entities hashtags
                hashtags = jdata["entities"]["hashtags"]
                hashtag_list = []
                for hashtag in hashtags:
                    hashtag_list.append(hashtag["text"].lower())

                # entities urls
                urls = jdata["entities"]["urls"]
                url_list = []
                for url in urls:
                    url_list.append(url["expanded_url"])

                # entities symbols
                symbols = jdata["entities"]["symbols"]
                symbols_list = []
                for symbol in symbols:
                    symbols_list.append(symbol["text"])

                # entities user_mentions
                user_mentions = jdata["entities"]["user_mentions"]
                user_mention_list = []
                for um in user_mentions:
                    user_mention_list.append(um["id"])

                # quoted status id if exists
                if "quoted_status_id" in jdata:
                    quoted_status_id = jdata["quoted_status_id"]
                else:
                    quoted_status_id = None

                # place exists
                place = jdata["place"]
                if place is not None:
                    place_full_name = place["full_name"]
                    place_coordinates = place['bounding_box']['coordinates'][0][0]
                else:
                    place_full_name = None
                    place_coordinates = None

                coordinates = jdata["coordinates"]
                # user_location, only compute geocode if other means have failed
                geocode_coordinates_of_user_location = []
                str_user_loc = jdata["user"]["location"]
                if str_user_loc is not None and "," in str_user_loc:
                    str_user_loc = str_user_loc.split(",")[0].strip()
                if str_user_loc is not None and len(
                        str_user_loc) < 25 and coordinates is None and place_full_name is None:
                    geocode_obj = None
                    if str_user_loc in LOCATION_COORDINATES.keys():
                        geocode_obj = LOCATION_COORDINATES[str_user_loc]
                    else:
                        # geocode_obj=None #currently the api for getting geo codes seems to be unstable
                        try:
                            geocode_obj = geolocator.geocode(str_user_loc)
                            LOCATION_COORDINATES[str_user_loc] = geocode_obj
                        except Exception as exc:
                            traceback.print_exc(file=sys.stdout)
                            logger.error("geopy not responding to: {}".format(str_user_loc))

                    if geocode_obj is not None:
                        geocode_coordinates_of_user_location.append(geocode_obj.latitude)
                        geocode_coordinates_of_user_location.append(geocode_obj.longitude)

                # ml_tag=util.ml_tag(jdata['text'], feat_vectorizer,self.__ml_model, self.__selected_features,
                #                    SCALING_STRATEGY, self.__sysout, logger)
                ml_tag = '0' if random.random() < 0.2 else '2'
                tweet_risk = random.uniform(0, 1.0)


                if coordinates==None:
                    coordinates=place_coordinates
                if coordinates==None:
                    coordinates=geocode_coordinates_of_user_location

                docs = [{'id': jdata["id"],
                         'created_at': str_solr_time,
                         'coordinates': coordinates,
                         'favorite_count': jdata["favorite_count"],
                         'in_reply_to_screen_name': jdata["in_reply_to_screen_name"],
                         'in_reply_to_status_id': jdata["in_reply_to_status_id"],
                         'in_reply_to_user_id': jdata["in_reply_to_user_id"],
                         'lang': jdata["lang"],
                         'place_full_name': place_full_name,
                         'place_coordinates': place_coordinates,
                         'retweet_count': jdata["retweet_count"],
                         'retweeted': jdata["retweeted"],
                         'quoted_status_id': quoted_status_id,
                         'status_text': jdata["text"],
                         'entities_hashtag': hashtag_list,
                         'entities_symbol': symbols_list,
                         'entities_url': url_list,
                         'entities_user_mention': user_mention_list,
                         'user_id': jdata["user"]["id"],
                         'user_screen_name': jdata["user"]["screen_name"],
                         'user_statuses_count': jdata["user"]["statuses_count"],
                         'user_friends_count': jdata["user"]["friends_count"],
                         'user_followers_count': jdata["user"]["followers_count"],
                         'user_location': str_user_loc,
                         'user_location_coordinates': geocode_coordinates_of_user_location,
                         'ml_tag': ml_tag,
                         "tweet_risk": tweet_risk}]
                self.__solr.index(self.__core, docs)
        except Exception as exc:
            traceback.print_exc(file=sys.stdout)
            print("Error encountered for {}, error:{} (see log file for details)".format(self.__count, exc))
            if jdata is not None and "id" in jdata.keys():
                tweet_id = jdata["id"]
            else:
                tweet_id = "[failed to parse]"
            logger.info("Error encountered for counter={}, tweet={}, error:{} (see log file for details)".
                        format(self.__count, tweet_id, exc))
            if jdata is not None:
                file = LOG_DIR + "/" + str(tweet_id) + ".txt"
                logger.info("\t input data json written to {}".format(file))
                with open(file, 'w') as outfile:
                    json.dump(jdata, outfile)
            pass
        return (True)

    def on_error(self, status):
        print(status)

    def on_status(self, status):
        print(status.text)


oauth = read_oauth(sys.argv[1])
print(sys.argv[1])
sc = read_search_criteria(sys.argv[2])
print(sys.argv[2])
auth = OAuthHandler(oauth["C_KEY"], oauth["C_SECRET"])
auth.set_access_token(oauth["A_TOKEN"], oauth["A_SECRET"])
twitterStream = Stream(auth, TwitterStream(sys.argv[3], sys.argv[4], sys.argv[5]))
twitterStream.filter(track=[sc["KEYWORDS"]], languages=LANGUAGES_ACCETED)

# searcher = TwitterSearch(auth)
# searcher.index(["#refugeesnotwelcome","#DeportallMuslims", "#banislam","#banmuslims", "#destroyislam",
#                 "#norefugees","#nomuslims"])
