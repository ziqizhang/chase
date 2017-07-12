#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import datetime
import pickle
import sys

import numpy
import pandas as pd
from sklearn.model_selection import train_test_split

from exp import experiment_settings
from ml import classifier_gridsearch as cl
from ml import text_preprocess as tp
from ml import util
from ml.vectorizer import feature_vectorizer as fv

# Model selection
WITH_SGD = True
WITH_SLR = True
WITH_RANDOM_FOREST = False #this algorithm may not work on very small feature vectors
WITH_LIBLINEAR_SVM = True
WITH_RBF_SVM = False
WITH_ANN = False

# feature scaling with bound [0,1] is ncessarily for MNB model
SCALING_STRATEGY_MIN_MAX = 0
# MEAN and Standard Deviation scaling is the standard feature scaling method
SCALING_STRATEGY_MEAN_STD = 1
SCALING_STRATEGY = 3

# DIRECTLY LOAD PRE-TRAINED MODEL FOR PREDICTION
# ENABLE THIS VARIABLE TO TEST NEW TEST SET WITHOUT TRAINING
LOAD_MODEL_FROM_FILE = False
# The number of CPUs to use to do the computation. -1 means 'all CPUs'
NUM_CPU = -1
N_FOLD_VALIDATION = 5
TEST_SPLIT_PERCENT = 0.25
DNN_FEATURE_SIZE = 300


#####################################################


class ChaseGridSearch(object):
    """
    """

    def __init__(self, task, identifier,
                 data_file,
                 feat_v: fv.FeatureVectorizer,
                 cl_gridsearch:bool,
                 dr_option, #feature optimization option, see create_feature_selector in classifier_trian
                 dr_gridsearch:bool, #if feature selection is used, whether to do grid search on the selector
                 fs_option,
                 fs_gridsearch:bool,
                 folder_sysout):
        self.raw_data = numpy.empty
        self.data_file = data_file
        self.identifier = identifier
        self.task_name = task
        self.feat_v = feat_v  # inclusive
        self.sys_out = folder_sysout  # exclusive 16
        self.feature_size = DNN_FEATURE_SIZE
        self.cl_gridsearch=cl_gridsearch
        self.dr_option=dr_option
        self.dr_gridsearch=dr_gridsearch
        self.fs_option=fs_option
        self.fs_gridsearch=fs_gridsearch

    def load_data(self):
        self.raw_data = pd.read_csv(self.data_file, sep=',', encoding="utf-8")

    def gridsearch(self):
        meta_M=util.feature_extraction(self.raw_data.tweet, self.feat_v, self.sys_out)
        M=meta_M[0]
        #M=self.feature_scale(M)

        # split the dataset into two parts, 0.75 for train and 0.25 for testing
        X_train_data, X_test_data, y_train, y_test = \
            train_test_split(M, self.raw_data['class'],
                             test_size=TEST_SPLIT_PERCENT,
                             random_state=42)
        X_train_data=util.feature_scale(X_train_data)
        X_test_data = util.feature_scale(X_test_data)
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

        # #if not self.feature_selection:
        # print("APPLYING FEATURE SCALING: [%s]" % SCALING_STRATEGY)
        #
        # if SCALING_STRATEGY == SCALING_STRATEGY_MEAN_STD:
        #     X_train_data = util.feature_scaling_mean_std(X_train_data)
        #     X_test_data = util.feature_scaling_mean_std(X_test_data)
        # elif SCALING_STRATEGY == SCALING_STRATEGY_MIN_MAX:
        #     X_train_data = util.feature_scaling_min_max(X_train_data)
        #     X_test_data = util.feature_scaling_min_max(X_test_data)
        # else:
        #     raise ArithmeticError("SCALING STRATEGY IS NOT SET CORRECTLY!")


        ######################### SGDClassifier #######################
        if WITH_SGD:
            cl.learn_general(NUM_CPU, N_FOLD_VALIDATION, self.task_name, LOAD_MODEL_FROM_FILE, "sgd",
                             meta_M[1], X_train_data, y_train,
                             X_test_data, y_test, self.identifier, self.sys_out,
                             self.cl_gridsearch, self.dr_option, self.dr_gridsearch,
                             self.fs_option,self.fs_gridsearch)

        ######################### Stochastic Logistic Regression#######################
        if WITH_SLR:
            cl.learn_general(NUM_CPU, N_FOLD_VALIDATION, self.task_name, LOAD_MODEL_FROM_FILE, "lr",
                             meta_M[1],X_train_data, y_train,
                             X_test_data, y_test, self.identifier, self.sys_out, self.cl_gridsearch
                             , self.dr_option, self.dr_gridsearch,
                             self.fs_option,self.fs_gridsearch)

        ######################### Random Forest Classifier #######################
        if WITH_RANDOM_FOREST:
            cl.learn_general(NUM_CPU, N_FOLD_VALIDATION, self.task_name, LOAD_MODEL_FROM_FILE, "rf",
                             meta_M[1],
                             X_train_data,
                             y_train,
                             X_test_data, y_test, self.identifier, self.sys_out, self.cl_gridsearch
                             , self.dr_option, self.dr_gridsearch,
                             self.fs_option,self.fs_gridsearch)

        ###################  liblinear SVM ##############################
        if WITH_LIBLINEAR_SVM:
            cl.learn_general(NUM_CPU, N_FOLD_VALIDATION, self.task_name, LOAD_MODEL_FROM_FILE,
                                    "svm-l",
                             meta_M[1],X_train_data,
                             y_train, X_test_data, y_test, self.identifier, self.sys_out,
                             self.cl_gridsearch, self.dr_option, self.dr_gridsearch,
                             self.fs_option,self.fs_gridsearch)

        ##################### RBF svm #####################
        if WITH_RBF_SVM:
            cl.learn_general(NUM_CPU, N_FOLD_VALIDATION, self.task_name, LOAD_MODEL_FROM_FILE,
                                    "svm-rbf",
                             meta_M[1],
                             X_train_data,
                             y_train, X_test_data, y_test, self.identifier, self.sys_out,
                             self.cl_gridsearch, self.dr_option, self.dr_gridsearch,
                             self.fs_option,self.fs_gridsearch)

        ################# Artificial Neural Network #################
        if WITH_ANN:
            cl.learn_dnn(NUM_CPU, N_FOLD_VALIDATION, self.task_name, LOAD_MODEL_FROM_FILE, "ann",
                         self.feature_size,
                         meta_M[1],
                         X_train_data,
                         y_train, X_test_data, y_test, self.identifier, self.sys_out,
                         self.cl_gridsearch, self.dr_option, self.dr_gridsearch,
                             self.fs_option,self.fs_gridsearch)

        print("complete, {}".format(datetime.datetime.now()))


if __name__ == '__main__':
    settings = experiment_settings.create_settings(sys.argv[1], sys.argv[2])
    for ds in settings:
        print("##########\nSTARTING EXPERIMENT SETTING:" + '; '.join(map(str, ds)))
        classifier = ChaseGridSearch(ds[0], ds[1], ds[2], ds[3], ds[4], ds[5], ds[6], ds[7],
                                     ds[8], ds[9])
        classifier.load_data()

        # ============= random sampling =================================
        # print("training data size before resampling:", len(classifier.training_data))
        # X_resampled, y_resampled = classifier.under_sampling(classifier.training_data,                                                         classifier.training_label)
        # print("training data size after resampling:", len(X_resampled))
        # enable this line to visualise the data
        # classifier.training_data = X_resampled
        # classifier.training_label = y_resampled

        classifier.gridsearch()
        # classifier.testing([fe.NGRAM_FEATURES_VOCAB, fe.NGRAM_POS_FEATURES_VOCAB, fe.TWEET_TD_OTHER_FEATURES_VOCAB],
        #                    sys.argv[1]+"/features",
        #                    sys.argv[1])
