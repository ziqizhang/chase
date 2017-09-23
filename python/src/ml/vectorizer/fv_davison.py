import datetime
from ml import feature_extractor as fe
from ml import text_preprocess as tp
from ml import nlp
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from ml.vectorizer import feature_vectorizer as fv
from util import logger

class FeatureVectorizerDavidson(fv.FeatureVectorizer):
    def __init__(self):
        super().__init__()
        self.ngram_vectorizer = TfidfVectorizer(
            # vectorizer = sklearn.feature_extraction.text.CountVectorizer(
            tokenizer=nlp.tokenize,
            preprocessor=tp.preprocess,
            ngram_range=(1, 3),
            stop_words=nlp.stopwords,  # We do better when we keep stopwords
            use_idf=True,
            smooth_idf=False,
            norm=None,  # Applies l2 norm smoothing
            decode_error='replace',
            max_features=10000,
            min_df=5,
            max_df=0.501
        )
        self.pos_vectorizer = TfidfVectorizer(
            # vectorizer = sklearn.feature_extraction.text.CountVectorizer(
            tokenizer=None,
            lowercase=False,
            preprocessor=None,
            ngram_range=(1, 3),
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
        """
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
        #to use original DV preprocessing, replce tweets_cleaned with tweets_original
        tfidf = fe.get_ngram_tfidf(self.ngram_vectorizer, tweets_cleaned, out_folder, flag)

        # Features group 2: PoS for ngrams
        # Features group 2: PoS for ngrams
        pos=fe.get_ngram_pos_tfidf(self.pos_vectorizer, tweets_cleaned, out_folder, flag)

        # Features group 3: other features
        logger.logger.info("\tgenerating other feature vectors, {}".format(datetime.datetime.now()))
        feats = fe.get_oth_features(tweets_cleaned, tweets_cleaned,out_folder)
        logger.logger.info("\t\tcompleted, {}, {}".format(feats[0].shape,datetime.datetime.now()))

        # Now concatenate all features in to single sparse matrix
        M = np.concatenate([tfidf[0], pos[0], feats[0]], axis=1)
        #print(M.shape)
        features_by_type={}
        features_by_type[fe.NGRAM_FEATURES_VOCAB]=tfidf
        features_by_type[fe.NGRAM_POS_FEATURES_VOCAB]=pos
        features_by_type[fe.TWEET_TD_OTHER_FEATURES_VOCAB]=feats
        return [pd.DataFrame(M), features_by_type]
