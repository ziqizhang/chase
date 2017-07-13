#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import datetime
import sys
import os

import pandas as pd
from exp import exp_traintest
from ml import classifier_gridsearch as cg
from ml import classifier_traintest as ct
from ml import util
from ml.vectorizer import feature_vectorizer as fv

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

    def __init__(self,
                 task,
                 identifier,
                 data_train,
                 data_test,
                 feat_v: fv.FeatureVectorizer,
                 fs_option,
                 folder_sysout):
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
        self.load_data()
        meta_TRAIN = util.feature_extraction(self.raw_train.tweet, self.feat_v, self.sys_out)
        X_train = meta_TRAIN[0]
        X_train = util.feature_scale(SCALING_STRATEGY, X_train)
        y_train = self.raw_train['class'].astype(int)

        saved_feature_dir = self.sys_out + "/fs/"
        try:
            os.stat(saved_feature_dir)
        except:
            os.mkdir(saved_feature_dir)
        training_feature_save = saved_feature_dir + str(self.fs_option) + ".csv"

        select = cg.create_feature_selector(self.fs_option, False)[0]
        feature_selected = select is not None
        if feature_selected:
            X_train = select.fit_transform(X_train, y_train)
            feature_idx_stat = select.get_support()
            feature_idx = []
            for i, item in enumerate(feature_idx_stat):
                if item:
                    feature_idx.append(i)
            util.save_selected_features(feature_idx,
                                        meta_TRAIN[1],
                                        training_feature_save)
        else:
            feature_idx = []
            for i in range(0, len(X_train[0])):
                feature_idx.append(i)
            util.save_selected_features(feature_idx,
                                        meta_TRAIN[1],
                                        training_feature_save)
        print("FEATURE SELECTION={}, Shape={}".format(select, X_train.shape))

        X_test=self.transform_test_features(SCALING_STRATEGY,
                           self.feat_v,
                           training_feature_save,
                           self.sys_out)
        y_test = self.raw_test['class'].astype(int)

        ######################### SGDClassifier #######################
        if WITH_SGD:
            classifier = ct.create_classifier('sgd', self.sys_out, self.task_name, -1, 300)
            self.traintest(classifier[0], X_train, y_train,
                           X_test,y_test,
                           self.sys_out, 'sgd', self.task_name)

        ######################### Stochastic Logistic Regression#######################
        if WITH_SLR:
            classifier = ct.create_classifier('lr', self.sys_out, self.task_name, -1, 300)
            self.traintest(classifier[0], X_train, y_train,
                           X_test,y_test,
                           self.sys_out, 'lr', self.task_name)

        ######################### Random Forest Classifier #######################
        if WITH_RANDOM_FOREST:
            classifier = ct.create_classifier('rf', self.sys_out, self.task_name, -1, 300)
            self.traintest(classifier[0], X_train, y_train,
                           X_test,y_test,
                           self.sys_out, 'rf', self.task_name)

        ###################  liblinear SVM ##############################
        if WITH_LIBLINEAR_SVM:
            classifier = ct.create_classifier('svm-l', self.sys_out, self.task_name, -1, 300)
            self.traintest(classifier[0], X_train, y_train,
                           X_test,y_test,
                           self.sys_out, 'svm-l', self.task_name)

        ##################### RBF svm #####################
        if WITH_RBF_SVM:
            classifier = ct.create_classifier('svm-rbf', self.sys_out, self.task_name, -1, 300)
            self.traintest(classifier[0], X_train, y_train,
                           X_test,y_test,
                           self.sys_out, 'svm-rbf', self.task_name)

        ################# Artificial Neural Network #################
        if WITH_ANN:
            classifier = ct.create_classifier('ann', self.sys_out, self.task_name, -1, 300)
            self.traintest(classifier[0], X_train, y_train,
                           X_test,y_test,
                           self.sys_out, 'ann', self.task_name)

    def transform_test_features(self, scaling_option,
                                feat_vectorizer,
                                training_feature_save,
                                sys_out):
        X_test = ct.transform_test_features(self.raw_test.tweet,
                                            feat_vectorizer,
                                            training_feature_save,
                                            sys_out,
                                            scaling_option)
        return X_test

    def traintest(self, classifier, X_train, y_train,
                  X_test, y_test,
                  sys_out, model_name, identifier):
        classifier = classifier.fit(X_train, y_train)

        y_preds = classifier.predict(X_test)
        util.save_scores(y_preds, y_test, y_preds, y_test, model_name, self.task_name, identifier, 2, sys_out)
        print("complete, {}".format(datetime.datetime.now()))


if __name__ == '__main__':
    settings = exp_traintest.create_settings(sys.argv[1], sys.argv[2], sys.argv[3])
    for ds in settings:
        print("##########\nSTARTING EXPERIMENT SETTING:" + '; '.join(map(str, ds)))
        classifier = ChaseClassifier(ds[0],  # task
                                     ds[1],  # identifier
                                     ds[2],  # data train
                                     ds[3],  # data test
                                     ds[4],  # fv
                                     ds[5],  # fs option
                                     ds[6]  # outfolder
                                     )
        classifier.train_test()
