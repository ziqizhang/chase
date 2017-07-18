import datetime
import functools
import pickle

import enchant
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.util import skipgrams
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from textstat.textstat import *

from ml import nlp
from util import logger as ec

NGRAM_FEATURES_VOCAB="feature_vocab_ngram"
NGRAM_POS_FEATURES_VOCAB="feature_vocab_ngram_pos"
SKIPGRAM_FEATURES_VOCAB="feature_vocab_skipgram"
SKIPGRAM_POS_FEATURES_VOCAB="feature_vocab_skipgram_pos"
TWEET_TD_OTHER_FEATURES_VOCAB="feature_vocab_td_other"
TWEET_HASHTAG_FEATURES_VOCAB="feature_vocab_hashtag"
TWEET_CAPS_FEATURES_VOCAB="feature_vocab_capitalization"
TWEET_MISSPELLING_FEATURES_VOCAB="feature_vocab_misspelling"
TWEET_SPECIALPUNC_FEATURES_VOCAB="feature_vocab_special_punc"
TWEET_SPECIALCHAR_FEATURES_VOCAB="feature_vocab_special_char"
TWEET_DEPENDENCY_FEATURES_VOCAB="feature_vocab_dependency"

#generates tfidf weighted ngram feature as a matrix and the vocabulary
def get_ngram_tfidf(ngram_vectorizer: TfidfVectorizer, tweets, out_folder, flag):
    joblib.dump(ngram_vectorizer, out_folder + '/'+flag+'_ngram_tfidf.pkl')
    ec.logger.info("\tgenerating n-gram vectors, {}".format(datetime.datetime.now()))
    tfidf = ngram_vectorizer.fit_transform(tweets).toarray()
    ec.logger.info("\t\t complete, dim={}, {}".format(tfidf.shape,datetime.datetime.now()))
    vocab = {v: i for i, v in enumerate(ngram_vectorizer.get_feature_names())}
    idf_vals = ngram_vectorizer.idf_
    idf_dict = {i: idf_vals[i] for i in vocab.values()}  # keys are indices; values are IDF scores
    pickle.dump(vocab, open(out_folder+"/"+NGRAM_FEATURES_VOCAB+".pk", "wb" ))
    return tfidf, vocab


#generates tfidf weighted PoS of ngrams as a feature matrix and the vocabulary
def get_ngram_pos_tfidf(pos_vectorizer:TfidfVectorizer, tweets, out_folder, flag):
    ec.logger.info("\tcreating pos tags, {}".format(datetime.datetime.now()))
    tweet_tags = nlp.get_pos_tags(tweets)
    ec.logger.info("\tgenerating pos tag vectors, {}".format(datetime.datetime.now()))
    pos = pos_vectorizer.fit_transform(pd.Series(tweet_tags)).toarray()
    joblib.dump(pos_vectorizer, out_folder + '/'+flag+'_pos.pkl')
    ec.logger.info("\t\tcompleted, dim={}, {}".format(pos.shape,datetime.datetime.now()))
    pos_vocab = {v: i for i, v in enumerate(pos_vectorizer.get_feature_names())}
    pickle.dump(pos_vocab, open(out_folder+"/"+NGRAM_POS_FEATURES_VOCAB+".pk", "wb" ))
    return pos, pos_vocab

def get_skipgram(cleaned_tweets, out_folder, nIn, kIn):
    skipgram_feature_matrix = []
    skipper = functools.partial(skipgrams, n=nIn, k=kIn)

    for t in cleaned_tweets:
        tweetTokens = word_tokenize(t)
        skipgram_feature_matrix.append(list(skipper(tweetTokens)))
    return skipgram_feature_matrix


#todo: return a hashtag matrix indicating the presence of particular hashtags in tweets. input is the list of all tweets
#DictVectorizer should be used, see http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html#sklearn.feature_extraction.DictVectorizer
def get_hashtags_in_tweets(tweets, out_folder):
    hashtag_dict = {}
    hashtag_regex = '#[\w\-]+'

    count = 0

    for t in tweets:
        #temp = {}
        emoji_regex = '&#[0-9]{4,6};'
        t = re.sub(emoji_regex,'',t)
        while True:
            try:
                position = re.search(hashtag_regex, t)
                match = t[position.start():position.end()].lower()
                if position != None and match not in hashtag_dict.keys():
                    hashtag_dict[match] = count
                    t = t[position.end():]
                    count = count + 1
                else:
                    break
            except AttributeError:
                break
            except ValueError:
                break
    cv = CountVectorizer(vocabulary=hashtag_dict, token_pattern='\#\w+')
    hashtag_feature_matrix = cv.fit_transform(tweets)
    #import pandas as pd
    #pd.set_option('display.max_colwidth', -1)
    #pd.DataFrame(hashtag_feature_matrix.A, columns=cv.get_feature_names()).to_csv('outputdataframe.csv')
    hashtag_feature_vocab=None
    pickle.dump(hashtag_feature_vocab,
                open(out_folder+"/"+TWEET_HASHTAG_FEATURES_VOCAB+".pk", "wb" ))
    return hashtag_feature_matrix


#todo: return matrix containing a number indicating the extent to which CAPs are used in the tweets
#see 'other_features_' that processes a single tweet, and 'get_oth_features' that calls the former to process all tweets
def get_capitalization(tweets, cleaned_tweets,out_folder):
    caps_feature_matrix = []
    for t in tweets:
        totalChar = sum(1 for c in t if c != ' ')
        numcaps = sum(1 for c in t if c.isupper())
        caps_feature_matrix.append((numcaps/totalChar)*100)
    caps_feature_vocab="CAPITALIZATION"
    pickle.dump(caps_feature_vocab,
                open(out_folder+"/"+TWEET_CAPS_FEATURES_VOCAB+".pk", "wb"))
    return caps_feature_matrix
#todo: return matrix containing a number indicating the extent to which misspellings are found in the tweets
#see 'other_features_' that processes a single tweet, and 'get_oth_features' that calls the former to process all tweets
def get_misspellings(tweets, cleaned_tweets,out_folder):
    mispellings_feature_matrix = []
    #import time
    #start_time = time.time()
    d = enchant.Dict('en_UK')
    dus = enchant.Dict('en_US')
    for tweet in cleaned_tweets:
        totalchar = 0
        mispellings = 0
        tweet = re.sub(r'[^a-zA-Z\'\s]', '', tweet)
        # line = line.title()
        words = word_tokenize(tweet)

        for word in words:
            word = word[0].upper() + word[1:]
            if d.check(word) == False and re.match('\'', word) == None and dus.check(word) == False:
                #print(word)
                mispellings = mispellings + 1

        if len(words) != 0:
            mispellings_feature_matrix.append((mispellings/len(words))*100)
        else:
            mispellings_feature_matrix.append(0) #Line with only punctuation
    #print("--- %s seconds ---" % (time.time() - start_time))
    caps_feature_vocab="MISSPELLINGS"
    pickle.dump(caps_feature_vocab,
                open(out_folder+"/"+TWEET_MISSPELLING_FEATURES_VOCAB+".pk", "wb" ))
    return mispellings_feature_matrix

#todo: return matrix containing a number indicating the extent to which special chars are found in the tweets
#see 'other_features_' that processes a single tweet, and 'get_oth_features' that calls the former to process all tweets
def get_specialchars(tweets, cleaned_tweets,out_folder):
    specialchar_feature_matrix=[]
    for t in tweets:
        specialchar_feature_matrix.append(len(re.findall('&#[0-9]{4,6};', t))) #Emoji on twitter is handled by &# followed by 4-6 numbers and a ;
    specialchar_feature_vocab="SPECIALCHAR"
    pickle.dump(specialchar_feature_vocab,
                open(out_folder+"/"+TWEET_SPECIALCHAR_FEATURES_VOCAB+".pk", "wb" ))
    return specialchar_feature_matrix

#todo: return matrix containing a number indicating the extent to which special punctuations are found in the tweets
#see 'other_features_' that processes a single tweet, and 'get_oth_features' that calls the former to process all tweets
def get_specialpunct(tweets, cleaned_tweets,out_folder):
    specialpunc_feature_matrix=[]
    for t in cleaned_tweets:
        specialpunc_feature_matrix.append(len(re.findall('\!|\?', t)))
    specialpunc_feature_vocab = "SPECIALPUNC"
    pickle.dump(specialpunc_feature_vocab,
                open(out_folder+"/"+TWEET_SPECIALPUNC_FEATURES_VOCAB+".pk", "wb" ))
    return specialpunc_feature_matrix

#todo: this should encode 'we vs them' patterns in tweets but this is the most complicated..
def get_dependency_feature(tweets, cleaned_tweets,out_folder):
    dependency_feature_matrix=None
    dependency_feature_vocab=None
    pickle.dump(dependency_feature_vocab,
                open(out_folder+"/"+TWEET_DEPENDENCY_FEATURES_VOCAB+".pk", "wb" ))


def other_features_(tweet, cleaned_tweet):
    """This function takes a string and returns a list of features.
    These include Sentiment scores, Text and Readability scores,
    as well as Twitter specific features.

    This is modified to only include those features in the final
    model."""

    sentiment = nlp.sentiment_analyzer.polarity_scores(tweet)

    words = cleaned_tweet #Get text only

    syllables = textstat.syllable_count(words) #count syllables in words
    num_chars = sum(len(w) for w in words) #num chars in words
    num_chars_total = len(tweet)
    num_terms = len(tweet.split())
    num_words = len(words.split())
    avg_syl = round(float((syllables+0.001))/float(num_words+0.001),4)
    num_unique_terms = len(set(words.split()))
    ###Modified FK grade, where avg words per sentence is just num words/1
    FKRA = round(float(0.39 * float(num_words)/1.0) + float(11.8 * avg_syl) - 15.59,1)
    ##Modified FRE score, where sentence fixed to 1
    FRE = round(206.835 - 1.015*(float(num_words)/1.0) - (84.6*float(avg_syl)),2)


    twitter_objs = count_twitter_objs(tweet) #Count #, @, and http://
    features = [FKRA, FRE, syllables, num_chars, num_chars_total, num_terms, num_words,
                num_unique_terms, sentiment['compound'],
                twitter_objs[2], twitter_objs[1],]
    #features = pandas.DataFrame(features)
    return features



def count_twitter_objs(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE
    4) hashtags with HASHTAGHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned.

    Returns counts of urls, mentions, and hashtags.
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                       '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    hashtag_regex = '#[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
    #parsed_text = re.sub('#', '', parsed_text) #replace the tag leave the word
    parsed_text = re.sub(hashtag_regex, 'HASHTAGHERE', parsed_text)
    return (parsed_text.count('URLHERE'), parsed_text.count('MENTIONHERE'), parsed_text.count('HASHTAGHERE'))


def get_oth_features(tweets, cleaned_tweets,out_folder):
    """Takes a list of tweets, generates features for
    each tweet, and returns a numpy array of tweet x features"""
    feats=[]
    count=0
    # skipgram = get_skipgram(cleaned_tweets, out_folder, 2,2)
    # for line in skipgram:
    #     print(line)
    # hashtags = get_hashtags_in_tweets(tweets, out_folder)
    # mispellings = get_misspellings(tweets, cleaned_tweets, out_folder)
    # specialpunc = get_specialpunct(tweets, cleaned_tweets,out_folder)
    # specialchars = get_specialchars(tweets, cleaned_tweets,out_folder)
    # capitalization = get_capitalization(tweets,cleaned_tweets,out_folder)
    for t, tc in zip(tweets, cleaned_tweets):
        feats.append(other_features_(t, tc))
        count+=1
        # if count%100==0:
        #     print("\t {}".format(count))
    other_features_names = ["FKRA", "FRE","num_syllables", "num_chars", "num_chars_total",
                        "num_terms", "num_words", "num_unique_words", "vader compound",
                            "num_hashtags", "num_mentions"]
    feature_matrix=np.array(feats)
    pickle.dump(other_features_names,
                open(out_folder+"/"+TWEET_TD_OTHER_FEATURES_VOCAB+".pk", "wb" ))

    return feature_matrix, other_features_names
