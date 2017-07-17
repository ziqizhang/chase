import logging
import sys

import datetime
from keras.layers import Dense, Dropout, Embedding, Conv1D, MaxPooling1D, LSTM
from keras.models import Sequential
from sklearn.cross_validation import cross_val_predict, train_test_split
from ml import util
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from keras.wrappers.scikit_learn import KerasClassifier
import os
from exp import classifier_gridsearch_main as cgm
import pandas as pd
from ml.vectorizer import fv_davison

INPUT_DIM=500
logger = logging.getLogger(__name__)
LOG_DIR=os.getcwd()+"/logs"
logging.basicConfig(filename=LOG_DIR+'/training.log', level=logging.INFO, filemode='w')


def learn_dnn(cpus, nfold, task, load_model,X_train, y_train, X_test, y_test,
              identifier, outfolder):
    print("== Perform ANN ...")  # create model
    logger.info("== Perform ANN ...")
    subfolder=outfolder+"/models"
    try:
        os.stat(subfolder)
    except:
        os.mkdir(subfolder)
    model = KerasClassifier(build_fn=create_model, verbose=0)
    # define the grid search parameters
    batch_size = [10]
    epochs = [100]
    dropout = [0.3]
    fs=SelectKBest(k=INPUT_DIM, score_func=f_classif)
    param_grid = dict(dropout_rate=dropout, batch_size=batch_size, nb_epoch=epochs)

    _X_train=fs.fit_transform(X_train, y_train)
    _classifier =  GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=cpus,
                        cv=nfold)

    cv_score_ann = 0
    best_param_ann = []
    ann_model_file = os.path.join(subfolder, "ann-%s.m" % task)
    nfold_predictions=None

    if load_model:
        print("model is loaded from [%s]" % str(ann_model_file))
        best_estimator = util.load_classifier_model(ann_model_file)
    else:
        _classifier.fit(_X_train, y_train)
        nfold_predictions=cross_val_predict(_classifier.best_estimator_, _X_train, y_train, cv=nfold)

        cv_score_ann = _classifier.best_score_
        best_param_ann = _classifier.best_params_
        print("+ best params for {} model are:{}".format(model, best_param_ann))
        best_estimator = _classifier.best_estimator_

        #util.save_classifier_model(best_estimator, ann_model_file)

    print("testing on development set ....")
    X_test=None
    if(X_test is not None):
        heldout_predictions_final = best_estimator.predict(X_test)
        util.save_scores(nfold_predictions,y_train, heldout_predictions_final, y_test, 'dnn', task,
                         identifier,2,outfolder)

    else:
        util.save_scores(nfold_predictions,y_train, None, y_test, model, task, identifier,2,
                         outfolder)

    #util.print_eval_report(best_param_ann, cv_score_ann, dev_data_prediction_ann,
    #                       time_ann_predict_dev,
    #                       time_ann_train, y_test)

def create_model(dropout_rate=0.5):
    # create model
    # model = Sequential()
    # model.add(Dense(500, input_dim=1000, activation='relu'))
    # model.add(Dropout(dropout_rate))
    # model.add(Dense(1, activation='sigmoid'))
    # # Compile model
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


    model = Sequential()
    model.add(Embedding(input_dim=11230, output_dim=32, input_length=INPUT_DIM))
    model.add(Conv1D(filters=32, kernel_size=4, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("New run started at {}\n{}".format(datetime.datetime.now(),model.summary()))
    logger.info("New run started at {}\n{}".format(datetime.datetime.now(),model.summary()))
    return model


def gridsearch(data_file, feat_vectorizer, sys_out):
    raw_data = pd.read_csv(data_file, sep=',', encoding="utf-8")
    meta_M=util.feature_extraction(raw_data.tweet, feat_vectorizer, sys_out)
    M=meta_M[0]
    #M=self.feature_scale(M)

    # split the dataset into two parts, 0.75 for train and 0.25 for testing
    X_train_data, X_test_data, y_train, y_test = \
                train_test_split(M, raw_data['class'],
                                 test_size=cgm.TEST_SPLIT_PERCENT,
                                 random_state=42)
    X_train_data=util.feature_scale(cgm.SCALING_STRATEGY,X_train_data)
    X_test_data = util.feature_scale(cgm.SCALING_STRATEGY,X_test_data)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)


    learn_dnn(-1, cgm.N_FOLD_VALIDATION, 'td-original', cgm.LOAD_MODEL_FROM_FILE,
                             X_train_data,
                             y_train, X_test_data, y_test, "dense", sys_out)



gridsearch(sys.argv[1], fv_davison.FeatureVectorizerDavidson(), sys.argv[2])
