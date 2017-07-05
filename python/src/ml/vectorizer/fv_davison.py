import datetime
from sklearn.externals import joblib

from ml import feature_extractor as fe
from ml import text_preprocess as tp
from ml import nlp
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from ml.vectorizer import feature_vectorizer as fv

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
        joblib.dump(self.ngram_vectorizer, out_folder + '/'+flag+'_ngram_tfidf.pkl')
        print("\tgenerating n-gram vectors, {}".format(datetime.datetime.now()))
        tfidf = self.ngram_vectorizer.fit_transform(tweets_original).toarray()
        print("\t\t complete, dim={}, {}".format(tfidf.shape,datetime.datetime.now()))
        vocab = {v: i for i, v in enumerate(self.ngram_vectorizer.get_feature_names())}
        idf_vals = self.ngram_vectorizer.idf_
        idf_dict = {i: idf_vals[i] for i in vocab.values()}  # keys are indices; values are IDF scores

        # Features group 2: PoS for ngrams
        print("\tcreating pos tags, {}".format(datetime.datetime.now()))
        tweet_tags = nlp.get_pos_tags(tweets_cleaned)
        print("\tgenerating pos tag vectors, {}".format(datetime.datetime.now()))
        pos = self.pos_vectorizer.fit_transform(pd.Series(tweet_tags)).toarray()
        joblib.dump(self.pos_vectorizer, out_folder + '/'+flag+'_pos.pkl')
        print("\t\tcompleted, dim={}, {}".format(pos.shape,datetime.datetime.now()))
        pos_vocab = {v: i for i, v in enumerate(self.pos_vectorizer.get_feature_names())}

        # Features group 3: other features
        print("\tgenerating other feature vectors, {}".format(datetime.datetime.now()))
        feats = fe.get_oth_features(tweets_original, tweets_cleaned)
        print("\t\tcompleted, {}, {}".format(feats.shape,datetime.datetime.now()))

        # Now concatenate all features in to single sparse matrix
        M = np.concatenate([tfidf, pos, feats], axis=1)
        #print(M.shape)
        return pd.DataFrame(M)
