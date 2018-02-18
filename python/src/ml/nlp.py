import re

import nltk
from nltk import PorterStemmer, WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS

#sentiment_analyzer = VS()
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stopwords = nltk.corpus.stopwords.words("english")
other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)

# stem_or_lemma: 0 - apply porter's stemming; 1: apply lemmatization; 2: neither
# -set to 0 to reproduce Davidson. However, note that because a different stemmer is used, results could be
# sightly different
# -set to 2 will do 'basic_tokenize' as in Davidson
def tokenize(tweet, stem_or_lemma=0):
    """Removes punctuation & excess whitespace, sets to lowercase,
    and normalizes tweets. Returns a list of stemmed tokens."""
    tweet = " ".join(re.split("[^a-zA-Z]*", tweet.lower())).strip()
    if stem_or_lemma==0:
        tokens = [stemmer.stem(t) for t in tweet.split()]
    elif stem_or_lemma==1:
        tokens=[lemmatizer.lemmatize(t) for t in tweet.split()]
    else:
        tokens = [t for t in tweet.split()] #this is basic_tokenize in TD's original code
    return tokens


# tweets should have been preprocessed to the clean/right format before passing to this method
def get_pos_tags(tweets):
    """Takes a list of strings (tweets) and
    returns a list of strings of (POS tags).
    """
    tweet_tags = []
    for t in tweets:
        tokens = tokenize(t, 2)
        tags = nltk.pos_tag(tokens)
        tag_list = [x[1] for x in tags]
        #for i in range(0, len(tokens)):
        tag_str = " ".join(tag_list)
        tweet_tags.append(tag_str)
    return tweet_tags
