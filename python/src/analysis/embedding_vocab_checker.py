import functools
import re

import gensim
import pandas as pd
import logging
import pickle
import datetime
from ml import text_preprocess as tp
from sklearn.feature_extraction.text import CountVectorizer
from ml import nlp

logger = logging.getLogger(__name__)


def get_word_vocab(tweets, out_folder, normalize_option):
    word_vectorizer = CountVectorizer(
        # vectorizer = sklearn.feature_extraction.text.CountVectorizer(
        tokenizer=functools.partial(nlp.tokenize, stem_or_lemma=normalize_option),
        preprocessor=tp.strip_hashtags,
        ngram_range=(1, 1),
        stop_words=nlp.stopwords,  # We do better when we keep stopwords
        decode_error='replace',
        max_features=50000,
        min_df=1,
        max_df=0.999
    )

    logger.info("\tgenerating word vectors, {}".format(datetime.datetime.now()))
    counts = word_vectorizer.fit_transform(tweets).toarray()
    logger.info("\t\t complete, dim={}, {}".format(counts.shape, datetime.datetime.now()))
    vocab = {v: i for i, v in enumerate(word_vectorizer.get_feature_names())}
    pickle.dump(vocab, open(out_folder + "/" + "DNN_WORD_EMBEDDING" + ".pk", "wb"))

    word_embedding_input = []
    for tweet in counts:
        tweet_vocab = []
        for i in range(0, len(tweet)):
            if tweet[i] != 0:
                tweet_vocab.append(i)
        word_embedding_input.append(tweet_vocab)
    return word_embedding_input, vocab


def load_model(model_file):
    gensimFormat = ".gensim" in model_file
    if gensimFormat:
        model = gensim.models.KeyedVectors.load(model_file, mmap='r')
    else:
        model = gensim.models.KeyedVectors. \
            load_word2vec_format(model_file, binary=True)
    return model


def check_vocab(model, model_str, input_date_file, sys_out, word_norm_option):
    raw_data = pd.read_csv(input_date_file, sep=',', encoding="utf-8")
    M = get_word_vocab(raw_data.tweet, sys_out, word_norm_option)
    with open("oov_list", 'wb') as oov_file:
        oov = list()
        word_vocab = M[1]
        count = 0
        random = 0
        for word, i in word_vocab.items():
            nopunc = re.sub(r'[^\w\s]', '', word).strip()
            if len(nopunc) == 0:
                continue
            if len(word) == 1:
                continue
            if word in model.wv.vocab.keys():
                pass
            else:
                random += 1
                oov.append(word)
            count += 1
        print("data={}, model={}, norm={}, vocab={},oov={}".
              format(input_date_file, model_str, word_norm_option, count, random))
        sorted(oov)


def check_vocab_multi(models: list, input_date_file, sys_out, word_norm_option):
    raw_data = pd.read_csv(input_date_file, sep=',', encoding="utf-8")
    M = get_word_vocab(raw_data.tweet, sys_out, word_norm_option)

    word_vocab = M[1]
    count = 0
    random = 0
    for word, i in word_vocab.items():
        found = False
        for m in models:
            if word in m.wv.vocab.keys():
                found = True
                break
        if not found:
            random += 1
        count += 1
    print("data={}, model=all, norm={}, vocab={},oov={}".
          format(input_date_file, word_norm_option, count, random))


def check_vocab_presence(model_file, list: []):
    gensimFormat = ".gensim" in model_file
    if gensimFormat:
        model = gensim.models.KeyedVectors.load(model_file, mmap='r')
    else:
        model = gensim.models.KeyedVectors. \
            load_word2vec_format(model_file, binary=True)

    for word in list:
        if word in model.wv.vocab.keys():
            print(word + ", yes")
        else:
            print(word + ", no")


#emg_model = load_model("/home/zz/Work/data/GoogleNews-vectors-negative300.bin.gz")
#emt_model = load_model("/home/zz/Work/data/Set1_TweetDataWithoutSpam_Word.bin")
eml_model = load_model("/home/zz/Work/data/glove.840B.300d.bin.gensim")
input_data = [

    "/home/zz/Work/chase/data/ml/ml/dt/labeled_data_all_2.csvc.csv"]
    # "/home/zz/Work/chase/data/ml/ml/rm/labeled_data_all.csv",
    # "/home/zz/Work/chase/data/ml/ml/w/labeled_data_all.csv",
    # "/home/zz/Work/chase/data/ml/ml/w+ws/labeled_data_all.csv",
    # "/home/zz/Work/chase/data/ml/ml/ws-amt/labeled_data_all.csv",
    # "/home/zz/Work/chase/data/ml/ml/ws-exp/labeled_data_all.csv",
    # "/home/zz/Work/chase/data/ml/ml/ws-gb/labeled_data_all.csv"]
output = "/home/zz/Work/chase/output"

# list=["faggot"]
# check_vocab_presence(emg_model, list)

for input in input_data:
    # check_vocab(emg_model, 'google',
    #             input,
    #             output, 1)
    # check_vocab(emt_model, 'tw',
    #             input,
    #             output, 1)
    check_vocab(eml_model, 'glv',
                input,
                output, 1)
    # check_vocab_multi([emg_model, emt_model, eml_model], input, output, 1)

    # check_vocab(emg_model, 'google',
    #             input,
    #             output, 1)
    # check_vocab(emt_model, 'tw',
    #             input,
    #             output, 1)
    # check_vocab(eml_model, 'glv',
    #             input,
    #             output, 1)
    # check_vocab_multi([emg_model, emt_model, eml_model], input, output, 1)
