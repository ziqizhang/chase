'''everything is the same as chase_basic, but skip gram replaces ngram (skipgram is a superset)'''

import datetime
from ml import feature_extractor as fe
from ml import text_preprocess as tp
from ml import nlp
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from ml.vectorizer import feature_vectorizer as fv
from util import logger as logger

class FeatureVectorizerChaseSkipgram(fv.FeatureVectorizer):
    def __init__(self):
        super().__init__()
        self.unigram_vectorizer = TfidfVectorizer(
            # vectorizer = sklearn.feature_extraction.text.CountVectorizer(
            analyzer='word',
            tokenizer=nlp.tokenize,
            preprocessor=tp.preprocess,
            stop_words=nlp.stopwords,  # We do better when we keep stopwords
            use_idf=True,
            smooth_idf=False,
            norm=None,  # Applies l2 norm smoothing
            decode_error='replace',
            max_features=10000,
            min_df=5,
            max_df=0.501
        )
        self.unigram_pos_vectorizer = TfidfVectorizer(
            # vectorizer = sklearn.feature_extraction.text.CountVectorizer(
            tokenizer=None,
            lowercase=False,
            preprocessor=None,
            ngram_range=(1, 1),
            stop_words=None,  # We do better when we keep stopwords
            use_idf=False,
            smooth_idf=False,
            norm=None,  # Applies l2 norm smoothing
            decode_error='replace',
            max_features=5000,
            min_df=5,
            max_df=0.501,
        )

    def transform_inputs(self, tweets_original, tweets_cleaned, out_folder, flag):
        """"
        This function takes a list of tweets, along with used to
        transform the tweets into the format accepted by the model.

        Each tweet is decomposed into
        (a) An array of TF-IDF scores for a set of n-grams in the tweet.
        (b) An array of POS tag sequences in the tweet.
        (c) An array of features including sentiment, vocab, and readability.

        Returns a pandas dataframe where each row is the set of features
        for a tweet. The features are a subset selected using a Logistic
        Regression with L1-regularization on the training data.

        """
        # Features group 1: tfidf weighted n-grams
        unigram_tfidf = fe.get_ngram_tfidf(self.unigram_vectorizer, tweets_cleaned, out_folder, flag)

        # Features group 2: PoS for ngrams
        #unigram_pos=fe.get_ngram_pos_tfidf(self.unigram_pos_vectorizer, tweets_cleaned, out_folder, flag)

        # Features group 3: other features
        logger.logger.info("\tgenerating other feature vectors, {}".format(datetime.datetime.now()))
        td_otherfeats = fe.get_oth_features(tweets_original, tweets_cleaned,out_folder)

        '''CHASE skipgram'''
        tweet_tags = nlp.get_pos_tags(tweets_cleaned)

        logger.logger.info("\tgenerating CHASE 2 skip bigram feature vectors, {}".format(datetime.datetime.now()))
        c_skipgram_22=fe.get_skipgram(tweets_cleaned, out_folder, 2,2)
        logger.logger.info("\tgenerating CHASE 2 skip bigram pos feature vectors, {}".format(datetime.datetime.now()))
        c_bigram_pos=fe.get_skipgram(tweet_tags, out_folder, 2,2)

        logger.logger.info("\tgenerating CHASE 2 skip trigram feature vectors, {}".format(datetime.datetime.now()))
        c_skipgram_32=fe.get_skipgram(tweets_cleaned, out_folder, 3,2)
        logger.logger.info("\tgenerating CHASE 2 skip trigram pos feature vectors, {}".format(datetime.datetime.now()))
        c_trigram_pos=fe.get_skipgram(tweet_tags, out_folder, 3,2)

        '''CHASE basic features={}'''
        logger.logger.info("\tgenerating CHASE hashtag feature vectors, {}".format(datetime.datetime.now()))
        c_hashtags=fe.get_hashtags_in_tweets(tweets_original, out_folder)
        logger.logger.info("\tgenerating CHASE other stats feature vectors, {}".format(datetime.datetime.now()))
        c_stats=fe.get_chase_stats_features(tweets_original, tweets_cleaned, out_folder)


        logger.logger.info("\t\tcompleted, {}, {}".format(c_stats[0].shape,datetime.datetime.now()))

        # Now concatenate all features in to single sparse matrix
        M = np.concatenate([unigram_tfidf[0], #unigram_pos[0],
                            c_skipgram_22[0], c_skipgram_32[0],
                            c_bigram_pos[0], c_trigram_pos[0],
                            td_otherfeats[0],
                            c_hashtags[0],c_stats[0]], axis=1)
        #print(M.shape)
        features_by_type={}
        features_by_type[fe.NGRAM_FEATURES_VOCAB]=unigram_tfidf
        #features_by_type[fe.NGRAM_POS_FEATURES_VOCAB]=unigram_pos
        features_by_type[fe.SKIPGRAM22_FEATURES_VOCAB]=c_skipgram_22
        features_by_type[fe.SKIPGRAM22_POS_FEATURES_VOCAB]=c_bigram_pos
        features_by_type[fe.SKIPGRAM32_FEATURES_VOCAB]=c_skipgram_32
        features_by_type[fe.SKIPGRAM32_POS_FEATURES_VOCAB]=c_trigram_pos
        features_by_type[fe.TWEET_TD_OTHER_FEATURES_VOCAB]=td_otherfeats
        features_by_type[fe.TWEET_HASHTAG_FEATURES_VOCAB]=c_hashtags
        features_by_type[fe.TWEET_CHASE_STATS_FEATURES_VOCAB]=c_stats
        return [pd.DataFrame(M), features_by_type]
