#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import datetime
import sys

import numpy
import pandas as pd
from sklearn.metrics import classification_report

from exp import experiment_settings
from ml import classifier_gridsearch as cg
from ml import classifier_traintest as ct
from ml import util
from ml.vectorizer import feature_vectorizer as fv
from ml import feature_extractor as fe

# Model selection
WITH_SGD = True
WITH_SLR = True
WITH_RANDOM_FOREST = False  # this algorithm may not work on very small feature vectors
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
DNN_FEATURE_SIZE = 300


#####################################################


class ChaseClassifier(object):
    """
    """

    def __init__(self, task, identifier,
                 data_train,
                 data_test,
                 feat_v: fv.FeatureVectorizer,
                 fs_option,
                 folder_sysout):
        self.raw_data = numpy.empty
        self.data_train = data_train
        self.data_test = data_test
        self.identifier = identifier
        self.task_name = task
        self.feat_v = feat_v  # inclusive
        self.sys_out = folder_sysout  # exclusive 16
        self.feature_size = DNN_FEATURE_SIZE
        self.fs_option = fs_option

    def load_data(self):
        self.raw_train = pd.read_csv(self.data_train, sep=',', encoding="utf-8")
        self.raw_test = pd.read_csv(self.data_test, sep=',', encoding="utf-8")

    def train_test(self):
        meta_TRAIN = util.feature_extraction(self.raw_train.tweet, self.feat_v, self.sys_out)
        X_train = meta_TRAIN[0]
        X_train = util.feature_scale(X_train, SCALING_STRATEGY)
        y_train = self.raw_train['class'].astype(int)

        #todo:
        expected_feature_list=[]
        saved_feature_dir=''
        
        select = cg.create_feature_selector(self.fs_option, False)[0]
        if select is not None:
            X_train = select.fit_transform(X_train)
            # todo: save features

        ######################### SGDClassifier #######################
        if WITH_SGD:
            classifier = ct.create_classifier('sgd', self.sys_out, self.task_name, -1, 300)
            self.traintest(classifier, X_train, y_train,
                  select is not None, SCALING_STRATEGY,
                  self.feat_v,
                  expected_feature_list,
                  saved_feature_dir,
                  self.sys_out)

        ######################### Stochastic Logistic Regression#######################
        if WITH_SLR:
            classifier = ct.create_classifier('lr', self.sys_out, self.task_name, -1, 300)
            # todo: train and test using slr, save model and scores

        ######################### Random Forest Classifier #######################
        if WITH_RANDOM_FOREST:
            classifier = ct.create_classifier('rf', self.sys_out, self.task_name, -1, 300)
            # todo: train and test using rf, save model and scores

        ###################  liblinear SVM ##############################
        if WITH_LIBLINEAR_SVM:
            classifier = ct.create_classifier('svm-l', self.sys_out, self.task_name, -1, 300)
            # todo: train and test using svml, save model and scores

        ##################### RBF svm #####################
        if WITH_RBF_SVM:
            classifier = ct.create_classifier('svm-rbf', self.sys_out, self.task_name, -1, 300)
            # todo: train and test using svm rbf, save model and scores

        ################# Artificial Neural Network #################
        else:
            classifier = ct.create_classifier('ann', self.sys_out, self.task_name, -1, 300)
            # todo: train and test using ann, save model and scores


    def traintest(self, classifier, X_train, y_train,
                  feature_selected:bool, scaling_option,
                  feat_vectorizer,
                  expected_feature_list,
                  saved_feature_dir,
                  sys_out):
        classifier = classifier.fit(X_train, y_train)
        y_test = self.raw_test['class'].astype(int)
        X_test = ct.transform_test_features(self.raw_test.tweet,
                                            feat_vectorizer,
                                            expected_feature_list,
                                            saved_feature_dir,
                                            sys_out,
                                            feature_selected, scaling_option)

        y_preds = classifier.predict(X_test)
        #todo: save result
        report = classification_report(y_test, y_preds)
        print(report)
        print("complete, {}".format(datetime.datetime.now()))


if __name__ == '__main__':
    settings = experiment_settings.create_settings(sys.argv[1], sys.argv[2])
    for ds in settings:
        print("##########\nSTARTING EXPERIMENT SETTING:" + '; '.join(map(str, ds)))
        # classifier = ChaseClassifier(ds[0], ds[1], ds[2], ds[3], ds[4],ds[5],ds[6], ds[7],
        #                              ds[8],ds[9])
        # classifier.load_data()

        # ============= random sampling =================================
        # print("training data size before resampling:", len(classifier.training_data))
        # X_resampled, y_resampled = classifier.under_sampling(classifier.training_data,                                                         classifier.training_label)
        # print("training data size after resampling:", len(X_resampled))
        # enable this line to visualise the data
        # classifier.training_data = X_resampled
        # classifier.training_label = y_resampled

        # classifier.gridsearch()
        # classifier.testing([fe.NGRAM_FEATURES_VOCAB, fe.NGRAM_POS_FEATURES_VOCAB, fe.TWEET_TD_OTHER_FEATURES_VOCAB],
        #                    sys.argv[1]+"/features",
        #                    sys.argv[1])
