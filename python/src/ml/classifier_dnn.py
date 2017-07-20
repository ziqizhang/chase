import datetime
import logging
import os
import sys

import functools
import pandas as pd
import pickle
from keras.layers import Dense, Embedding, Conv1D, MaxPooling1D, LSTM, Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import cross_val_predict, train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from keras.preprocessing import sequence

from ml import util
from ml import nlp
from ml import text_preprocess as tp

WORD_EMBEDDING_DIM_INPUT = 100
WORD_EMBEDDING_DIM_OUTPUT = 50
logger = logging.getLogger(__name__)
LOG_DIR = os.getcwd() + "/logs"
logging.basicConfig(filename=LOG_DIR + '/dnn-log.txt', level=logging.INFO, filemode='w')


def get_dnn_wordembedding_input(tweets, out_folder):
    word_vectorizer = CountVectorizer(
        # vectorizer = sklearn.feature_extraction.text.CountVectorizer(
        tokenizer=nlp.tokenize,
        preprocessor=tp.preprocess,
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




def create_model(max_index=100):
    # create model
    model = Sequential()
    # model.add(Embedding(input_dim=max_index, output_dim=WORD_EMBEDDING_DIM_OUTPUT,
    #                      input_length=WORD_EMBEDDING_DIM_INPUT))
    # model.add(Dense(200, input_dim=WORD_EMBEDDING_DIM_OUTPUT, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1, activation='sigmoid'))
    # # Compile model
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model = Sequential()
    model.add(Embedding(input_dim=max_index, output_dim=WORD_EMBEDDING_DIM_OUTPUT,
                        input_length=WORD_EMBEDDING_DIM_INPUT))
    model.add(LSTM(100, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # model = Sequential()
    # model.add(Embedding(input_dim=max_index, output_dim=WORD_EMBEDDING_DIM_OUTPUT,
    #                     input_length=WORD_EMBEDDING_DIM_INPUT))
    # model.add(Conv1D(filters=100, kernel_size=4, padding='same', activation='relu'))
    # model.add(MaxPooling1D(pool_size=2))
    # model.add(LSTM(100, activation='tanh'))
    # model.add(Dense(1, activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #
    logger.info("New run started at {}\n{}".format(datetime.datetime.now(), model.summary()))
    return model


def learn_dnn(cpus, nfold, task, load_model, X_train, y_train, X_test, y_test,
              identifier, outfolder, embedding_layer_max_index,
              instance_data_source_tags=None, accepted_ds_tags: list = None):
    logger.info("== Perform ANN ...")
    subfolder = outfolder + "/models"
    try:
        os.stat(subfolder)
    except:
        os.mkdir(subfolder)

    create_model_with_args = functools.partial(create_model, max_index=embedding_layer_max_index)
    model = KerasClassifier(build_fn=create_model_with_args, verbose=0)
    # define the grid search parameters
    batch_size = [50, 100]
    epochs = [3, 5]
    param_grid = dict(batch_size=batch_size, nb_epoch=epochs)

    _classifier = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=cpus,
                               cv=nfold)

    cv_score_ann = 0
    best_param_ann = []
    ann_model_file = os.path.join(subfolder, "ann-%s.m" % task)
    nfold_predictions = None

    if load_model:
        logger.info("model is loaded from [%s]" % str(ann_model_file))
        best_estimator = util.load_classifier_model(ann_model_file)
    else:
        _classifier.fit(X_train, y_train)
        nfold_predictions = cross_val_predict(_classifier.best_estimator_, X_train, y_train, cv=nfold)

        cv_score_ann = _classifier.best_score_
        best_param_ann = _classifier.best_params_
        logger.info("+ best params for {} model are:{}".format(model, best_param_ann))
        best_estimator = _classifier.best_estimator_

        # util.save_classifier_model(best_estimator, ann_model_file)

    logger.info("testing on development set ....")
    if (X_test is not None):
        heldout_predictions_final = best_estimator.predict(X_test)
        util.save_scores(nfold_predictions, y_train, heldout_predictions_final, y_test, 'dnn', task,
                         identifier, 2, outfolder, instance_data_source_tags, accepted_ds_tags)

    else:
        util.save_scores(nfold_predictions, y_train, None, y_test, 'dnn', task, identifier, 2,
                         outfolder, instance_data_source_tags, accepted_ds_tags)

        # util.print_eval_report(best_param_ann, cv_score_ann, dev_data_prediction_ann,
        #                       time_ann_predict_dev,
        #                       time_ann_train, y_test)



def gridsearch(data_file, sys_out, output_scores_per_ds):
    raw_data = pd.read_csv(data_file, sep=',', encoding="utf-8")
    M = get_dnn_wordembedding_input(raw_data.tweet, sys_out)
    # M=self.feature_scale(M)
    M0 = M[0]

    # split the dataset into two parts, 0.75 for train and 0.25 for testing
    X_train_data, X_test_data, y_train, y_test = \
        train_test_split(M0, raw_data['class'],
                         test_size=0.25,
                         random_state=42)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    X_train_data = sequence.pad_sequences(X_train_data, maxlen=WORD_EMBEDDING_DIM_INPUT)
    X_test_data = sequence.pad_sequences(X_test_data, maxlen=WORD_EMBEDDING_DIM_INPUT)

    instance_data_source_column = None
    accepted_ds_tags = None
    if output_scores_per_ds:
        instance_data_source_column = pd.Series(raw_data.ds)
        accepted_ds_tags = ["c", "td"]

    learn_dnn(-1, 5, 'td-tdf', False,
              X_train_data,
              y_train, X_test_data, y_test, "dense", sys_out,
              len(M[1]), instance_data_source_column, accepted_ds_tags)

gridsearch(sys.argv[1],
           sys.argv[2], sys.argv[3])
