import os


os.environ['PYTHONHASHSEED'] = '0'
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed

set_random_seed(2)

import datetime
import logging
import sys
import functools
import gensim
import numpy
import random as rn

import pandas as pd
import pickle
from keras.layers import Embedding
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import cross_val_predict, train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from keras.preprocessing import sequence

from ml import util
from ml import nlp
from ml import text_preprocess as tp
from ml import dnn_model_creator as dmc

MAX_SEQUENCE_LENGTH = 100 #maximum # of words allowed in a tweet
WORD_EMBEDDING_DIM_OUTPUT = 300
logger = logging.getLogger(__name__)
LOG_DIR = os.getcwd() + "/logs"
logging.basicConfig(filename=LOG_DIR + '/dnn-log.txt', level=logging.INFO, filemode='w')


def get_word_vocab(tweets, out_folder, normalize_option):
    word_vectorizer = CountVectorizer(
        # vectorizer = sklearn.feature_extraction.text.CountVectorizer(
        tokenizer=functools.partial(nlp.tokenize, stem_or_lemma=normalize_option),
        preprocessor=tp.strip_hashtags,
        ngram_range=(1, 1),
        stop_words=nlp.stopwords,  # We do better when we keep stopwords
        decode_error='replace',
        max_features=10000,
        min_df=5,
        max_df=0.501
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


def create_model(model_descriptor:str, max_index=100, wemb_matrix=None):
    '''A model that uses word embeddings'''
    if wemb_matrix is None:
        embedding_layer = Embedding(input_dim=max_index, output_dim=WORD_EMBEDDING_DIM_OUTPUT,
                                    input_length=MAX_SEQUENCE_LENGTH)
    else:
        # load pre-trained word embeddings into an Embedding layer
        # note that we set trainable = False so as to keep the embeddings fixed
        embedding_layer = Embedding(input_dim=max_index, output_dim=len(wemb_matrix[0]),
                                    weights=[wemb_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=False)
    if model_descriptor.startswith("b_"):
        model_descriptor=model_descriptor[2:].strip()
        model=dmc.create_model_with_branch(embedding_layer, model_descriptor)
    elif model_descriptor.startswith("f_"):
        model=dmc.create_final_model_with_concat_cnn(embedding_layer,model_descriptor)
    else:
        model=dmc.create_model_without_branch(embedding_layer, model_descriptor)
    #create_model_conv_lstm_multi_filter(embedding_layer)

    logger.info("New run started at {}\n{}".format(datetime.datetime.now(), model.summary()))
    return model


# def create_model_lstm(embedding_layer):#start from simple model
#     # use features from svm to train dnn, do not use lstm
#     # features from svm, apply autoencoder to compress..
#     model = Sequential()
#     model.add(embedding_layer)
#     model.add(Dropout(0.2))
#     model.add(LSTM(units=100, return_sequences=True)) #continuous time vs discrete time, gating, forgetting,
#     model.add(GlobalMaxPooling1D())
#     # lstm params,lstm hidden layers which ones are feeding into
#     model.add(Dropout(0.2))
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     logger.info("New run started at {}\n{}".format(datetime.datetime.now(), model.summary()))
#     return model


# def create_model_conv_lstm(embedding_layer):
#     model = Sequential()
#     model.add(embedding_layer)
#     model.add(Dropout(0.2))
#     model.add(Conv1D(filters=100, kernel_size=4, padding='same', activation='relu'))
#     model.add(MaxPooling1D(pool_size=4))
#     model.add(LSTM(units=100, return_sequences=True))
#     model.add(GlobalMaxPooling1D())
#     #model.add(Dropout(0.2))
#     model.add(Dense(2, activation='softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
#     logger.info("New run started at {}\n{}".format(datetime.datetime.now(), model.summary()))
#     return model


#using deprecated Merge, see https://statcompute.wordpress.com/2017/01/08/an-example-of-merge-layer-in-keras/
# def create_model_conv_lstm_multi_filter(embedding_layer):
#     flts=100
#     kernel_sizes=[2,3,4]
#     submodels = []
#     for kw in kernel_sizes:    # kernel sizes
#        submodel = Sequential()
#         submodel.add(embedding_layer)
#         submodel.add(Dropout(0.2))
#         submodel.add(Conv1D(filters=flts,
#                             kernel_size=kw,
#                             padding='same',
#                             activation='relu'))
#         submodel.add(MaxPooling1D(pool_size=kw))
#         submodels.append(submodel)
#
#     big_model = Sequential()
#     #concat = Concatenate(axis=1)     #line B
#     #big_model.add(concat(submodels)) #line C
#     big_model.add(Merge(submodels, mode="concat", concat_axis=1))  #line A
#     big_model.add(LSTM(units=100, return_sequences=True))
#     big_model.add(GlobalMaxPooling1D())
#     #big_model.add(Dropout(0.2))
#     big_model.add(Dense(2, activation='softmax'))
#     big_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return big_model

#workaround for functional api,
#https://stackoverflow.com/questions/43151775/how-to-have-parallel-convolutional-layers-in-keras
# def create_model_conv_lstm_multi_filter(embedding_layer):
#     "sub_conv[2,3,4](dropout=0.2,conv1d=100-v,)"
#     flts=100
#     kernel_sizes=[2,3,4]
#     submodels = []
#     for kw in kernel_sizes:    # kernel sizes
#         submodel = Sequential()
#         submodel.add(embedding_layer)
#         submodel.add(Dropout(0.2))
#         submodel.add(Conv1D(filters=flts,
#                             kernel_size=kw,
#                             padding='same',
#                             activation='relu'))
#         submodel.add(MaxPooling1D(pool_size=kw))
#         # submodel.add(LSTM(units=100, return_sequences=True))
#         # submodel.add(GlobalMaxPooling1D())
#         submodels.append(submodel)
#
#     submodel_outputs = [model.output for model in submodels]
#     x = Concatenate(axis=1)(submodel_outputs)
#
#     parallel_layers=Model(inputs=embedding_layer.input, outputs=x)
#     print("submodel:")
#     parallel_layers.summary()
#     print("")
#     big_model = Sequential()
#     big_model.add(parallel_layers)
#     big_model.add(LSTM(units=100, return_sequences=True))
#     big_model.add(GlobalMaxPooling1D())
#     #big_model.add(Dropout(0.2))
#     big_model.add(Dense(2, activation='softmax'))
#
#     big_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     big_model.summary()
#
#     return big_model


class MyKerasClassifier(KerasClassifier):
    def predict(self, x, **kwargs):
        kwargs = self.filter_sk_params(self.model.predict, kwargs)
        proba = self.model.predict(x, **kwargs)
        if proba.shape[-1] > 1:
            classes = proba.argmax(axis=-1)
        else:
            classes = (proba > 0.5).astype('int32')
        return self.classes_[classes]


def pretrained_embedding(word_vocab: dict, embedding_model, expected_emb_dim, randomize_strategy):
    logger.info("loading pre-trained embedding model... {}".format(datetime.datetime.now()))
    model = embedding_model
    logger.info("loading complete. {}".format(datetime.datetime.now()))

    matrix = numpy.zeros((len(word_vocab), expected_emb_dim))
    count = 0
    random = 0
    for word, i in word_vocab.items():
        if word in model.wv.vocab.keys():
            vec = model.wv[word]
            matrix[i] = vec
        else:
            random += 1
            if randomize_strategy == 1:  # randomly set values following a continuous uniform distribution
                vec = numpy.random.random_sample(expected_emb_dim)
                matrix[i] = vec
            elif randomize_strategy == 2:  # randomly take a vector from the model
                max = len(model.wv.vocab.keys()) - 1
                index = rn.randint(0, max)
                word = model.index2word[index]
                vec = model.wv[word]
                matrix[i] = vec
        count += 1
        if count % 100 == 0:
            print(count)
    model = None
    print("randomized={}".format(random))
    return matrix


def learn_dnn(dataset_name, outfolder, model_descriptor:str,
              cpus, nfold, X_train, y_train, X_test, y_test,
              embedding_layer_max_index, pretrained_embedding_matrix=None,
              instance_data_source_tags=None, accepted_ds_tags: list = None):
    print("== Perform ANN ...")
    subfolder = outfolder + "/models"
    try:
        os.stat(subfolder)
    except:
        os.mkdir(subfolder)

    create_model_with_args = \
        functools.partial(create_model, max_index=embedding_layer_max_index,
                          wemb_matrix=pretrained_embedding_matrix,
                          model_descriptor=model_descriptor)
    #model = MyKerasClassifier(build_fn=create_model_with_args, verbose=0)
    model = KerasClassifier(build_fn=create_model_with_args, verbose=0)
    # define the grid search parameters
    batch_size = [100]
    epochs = [10]
    param_grid = dict(batch_size=batch_size, nb_epoch=epochs)

    _classifier = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=cpus,
                               cv=nfold)

    print("fitting model...")
    _classifier.fit(X_train, y_train)
    nfold_predictions = cross_val_predict(_classifier.best_estimator_, X_train, y_train, cv=nfold)

    best_param_ann = _classifier.best_params_
    logger.info("+ best params for {} model are:{}".format(model, best_param_ann))
    best_estimator = _classifier.best_estimator_

    # util.save_classifier_model(best_estimator, ann_model_file)

    logger.info("testing on development set ....")
    if (X_test is not None):
        heldout_predictions_final = best_estimator.predict(X_test)
        util.save_scores(nfold_predictions, y_train, heldout_predictions_final, y_test,
                         model_descriptor, dataset_name,
                         3, outfolder, instance_data_source_tags, accepted_ds_tags)

    else:
        util.save_scores(nfold_predictions, y_train, None, y_test,
                         model_descriptor, dataset_name, 3,
                         outfolder, instance_data_source_tags, accepted_ds_tags)

        # util.print_eval_report(best_param_ann, cv_score_ann, dev_data_prediction_ann,
        #                       time_ann_predict_dev,
        #                       time_ann_train, y_test)

def gridsearch(input_data_file, dataset_name, sys_out, model_descriptor:str,
               print_scores_per_class,
               word_norm_option,
               randomize_strategy,
               pretrained_embedding_model=None, expected_embedding_dim=None):
    raw_data = pd.read_csv(input_data_file, sep=',', encoding="utf-8")
    M = get_word_vocab(raw_data.tweet, sys_out, word_norm_option)
    # M=self.feature_scale(M)
    M0 = M[0]

    pretrained_word_matrix = None
    if pretrained_embedding_model is not None:
        pretrained_word_matrix = pretrained_embedding(M[1], pretrained_embedding_model, expected_embedding_dim,
                                                      randomize_strategy)

    # split the dataset into two parts, 0.75 for train and 0.25 for testing
    X_train_data, X_test_data, y_train, y_test = \
        train_test_split(M0, raw_data['class'],
                         test_size=0.25,
                         random_state=42)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    X_train_data = sequence.pad_sequences(X_train_data, maxlen=MAX_SEQUENCE_LENGTH)
    X_test_data = sequence.pad_sequences(X_test_data, maxlen=MAX_SEQUENCE_LENGTH)

    instance_data_source_column = None
    accepted_ds_tags = None
    if print_scores_per_class:
        instance_data_source_column = pd.Series(raw_data.ds)
        accepted_ds_tags = ["c", "td"]


    learn_dnn(dataset_name, sys_out, model_descriptor,
              -1, 5,
              X_train_data,
              y_train, X_test_data, y_test,
              len(M[1]), pretrained_word_matrix,
              instance_data_source_column, accepted_ds_tags)
    print("complete {}".format(datetime.datetime.now()))


##############################################
##############################################

#/home/zqz/Work/data/GoogleNews-vectors-negative300.bin.gz
# 300

emb_model = None
emb_dim=None
params={}

sys_argv=sys.argv
if len(sys.argv)==2:
    sys_argv= sys.argv[1].split(" ")

for arg in sys_argv:
    pv=arg.split("=",1)
    if(len(pv)==1):
        continue
    params[pv[0]]=pv[1]
if "scoreperclass" not in params.keys():
    params["scoreperclass"]=False
if "word_norm" not in params.keys():
    params["word_norm"]=0
if "oov_random" not in params.keys():
    params["oov_random"]=0
if "emb_model" in params.keys():
    emb_model = gensim.models.KeyedVectors. \
        load_word2vec_format(params["emb_model"], binary=True)
if "emb_dim" in params.keys():
    emb_dim=int(params["emb_dim"])

gridsearch(params["input"],
            params["dataset"],  # dataset name
            params["output"],  # output
            params["model_desc"], #model descriptor
            params["scoreperclass"], #print scores per class
            params["word_norm"], #0-stemming, 1-lemma, other-do nothing
            params["oov_random"], #0-ignore oov; 1-random init by uniform dist; 2-random from embedding
            emb_model,
            emb_dim)
sys.exit(0)


# input=/home/zqz/Work/chase/data/ml/ml/rm/labeled_data_all.csv
# output=/home/zqz/Work/chase/output
# dataset=rm
# model_desc="dropout=0.2,conv1d=100-4,maxpooling1d=4,lstm=100-True,gmaxpooling1d,dense=2-softmax"

