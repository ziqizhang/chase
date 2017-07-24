#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import datetime
import sys
import os

import numpy
import pandas as pd
from sklearn.model_selection import train_test_split

from exp import exp_gridsearch as exp
from ml import classifier_gridsearch as cl
from ml import util
from ml.vectorizer import feature_vectorizer as fv
from ml.vectorizer import fv_chase_basic
from ml.vectorizer import fv_chase_basic_othering
from ml.vectorizer import fv_chase_skipgram_pos_only
from ml.vectorizer import fv_chase_skipgram
from ml.vectorizer import fv_davison

from util import logger as ec

# Model selection
WITH_SGD = False
WITH_SLR = False
WITH_RANDOM_FOREST = False #this algorithm may not work on very small feature vectors
WITH_LIBLINEAR_SVM = True
WITH_RBF_SVM = False

# feature scaling with bound [0,1] is ncessarily for MNB model
SCALING_STRATEGY_MIN_MAX = 0
# MEAN and Standard Deviation scaling is the standard feature scaling method
SCALING_STRATEGY_MEAN_STD = 1
SCALING_STRATEGY = 0

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
                 folder_sysout,
                 output_scores_per_ds=False):
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
        self.output_scores_per_ds=output_scores_per_ds

    def load_data(self):
        ec.logger.info("loading input data from: {}, exist={}".format(self.data_file,
                                    os.path.exists(self.data_file)))
        self.raw_data = pd.read_csv(self.data_file, sep=',', encoding="utf-8")

    def gridsearch(self):
        meta_M=util.feature_extraction(self.raw_data.tweet, self.feat_v, self.sys_out,ec.logger)
        M=meta_M[0]
        #M=self.feature_scale(M)

        # split the dataset into two parts, 0.75 for train and 0.25 for testing
        X_train_data, X_test_data, y_train, y_test = \
            train_test_split(M, self.raw_data['class'],
                             test_size=TEST_SPLIT_PERCENT,
                             random_state=42)
        X_train_data=util.feature_scale(SCALING_STRATEGY,X_train_data)
        X_test_data = util.feature_scale(SCALING_STRATEGY,X_test_data)
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

        instance_data_source_column=None
        accepted_ds_tags=None
        if self.output_scores_per_ds:
            instance_data_source_column=pd.Series(self.raw_data.ds)
            accepted_ds_tags=["c","td"]

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
                             self.fs_option,self.fs_gridsearch,
                             instance_data_source_column, accepted_ds_tags)

        ######################### Stochastic Logistic Regression#######################
        if WITH_SLR:
            cl.learn_general(NUM_CPU, N_FOLD_VALIDATION, self.task_name, LOAD_MODEL_FROM_FILE, "lr",
                             meta_M[1],X_train_data, y_train,
                             X_test_data, y_test, self.identifier, self.sys_out, self.cl_gridsearch
                             , self.dr_option, self.dr_gridsearch,
                             self.fs_option,self.fs_gridsearch,
                             instance_data_source_column, accepted_ds_tags)

        ######################### Random Forest Classifier #######################
        if WITH_RANDOM_FOREST:
            cl.learn_general(NUM_CPU, N_FOLD_VALIDATION, self.task_name, LOAD_MODEL_FROM_FILE, "rf",
                             meta_M[1],
                             X_train_data,
                             y_train,
                             X_test_data, y_test, self.identifier, self.sys_out, self.cl_gridsearch
                             , self.dr_option, self.dr_gridsearch,
                             self.fs_option,self.fs_gridsearch,
                             instance_data_source_column, accepted_ds_tags)

        ###################  liblinear SVM ##############################
        if WITH_LIBLINEAR_SVM:
            cl.learn_general(NUM_CPU, N_FOLD_VALIDATION, self.task_name, LOAD_MODEL_FROM_FILE,
                                    "svml",
                             meta_M[1],X_train_data,
                             y_train, X_test_data, y_test, self.identifier, self.sys_out,
                             self.cl_gridsearch, self.dr_option, self.dr_gridsearch,
                             self.fs_option,self.fs_gridsearch,
                             instance_data_source_column, accepted_ds_tags)

        ##################### RBF svm #####################
        if WITH_RBF_SVM:
            cl.learn_general(NUM_CPU, N_FOLD_VALIDATION, self.task_name, LOAD_MODEL_FROM_FILE,
                                    "svmrbf",
                             meta_M[1],
                             X_train_data,
                             y_train, X_test_data, y_test, self.identifier, self.sys_out,
                             self.cl_gridsearch, self.dr_option, self.dr_gridsearch,
                             self.fs_option,self.fs_gridsearch,
                             instance_data_source_column, accepted_ds_tags)
        ec.logger.info("complete, {}".format(datetime.datetime.now()))

if __name__ == '__main__':
    #example: argv1=out folder, arvg2=input data csv file, arvg3=a label to indicate which
    # dataset and feature vectorizer is used,
    #e.g., 'c-tdf' for chase dataset using td feature; 'c-cbf' for chase dataset and chase basic feature
    #argv[4] - set to any non empty string if using any mixed dataset; otherwise False
    #argv[5] - 0 to use td orignal features; 1 to use chase-basic features

    fv=None
    if sys.argv[2]=='1':
        fv=fv_chase_basic.FeatureVectorizerChaseBasic()
    elif sys.argv[2]=='2':
        fv=fv_chase_skipgram.FeatureVectorizerChaseSkipgram()
    elif sys.argv[2]=='3':
        fv=fv_chase_skipgram_pos_only.FeatureVectorizerChaseOther()
    elif sys.argv[2]=='4':
        fv=fv_chase_basic_othering.FeatureVectorizerChaseBasicOthering()
    else:
        fv=fv_davison.FeatureVectorizerDavidson()

    fs_options=sys.argv[4].split(",")

    label_and_data=sys.argv[5].split(",")
    settings=[]
    for lad in label_and_data:
        l_d = lad.split("=")
        settings.extend(exp.create_settings(sys.argv[1], l_d[1], l_d[0], bool(sys.argv[2]),
                                   fv,fs_options))
    print("total settings={}".format(len(settings)))

    for ds in settings:
        ec.logger.info("\n##########\nSTARTING EXPERIMENT SETTING:" + '; '.join(map(str, ds)))
        classifier = ChaseGridSearch(ds[0], ds[1], ds[2], ds[3], ds[4], ds[5], ds[6], ds[7],
                                     ds[8], ds[9],ds[10])
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
