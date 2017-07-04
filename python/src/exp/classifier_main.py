#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os

import datetime

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

from exp import experiment_settings
from ml.vectorizer import feature_vectorizer as fv
from ml import classifier_train as cl
from ml import classifier_tag as ct
import numpy
from sklearn.model_selection import train_test_split
import pandas as pd
from ml import util
from ml import text_preprocess as tp

# Model selection
WITH_SGD = True
WITH_SLR = True
WITH_RANDOM_FOREST = True
WITH_LIBLINEAR_SVM = True
WITH_RBF_SVM = True
WITH_ANN = False

# Random Forest model(or any tree-based model) do not ncessarily need feature scaling
N_FOLD_VALIDATION_ONLY = True
SCALING = False
# feature scaling with bound [0,1] is ncessarily for MNB model
SCALING_STRATEGY_MIN_MAX = 0
# MEAN and Standard Deviation scaling is the standard feature scaling method
SCALING_STRATEGY_MEAN_STD = 1
SCALING_STRATEGY = SCALING_STRATEGY_MEAN_STD

# DIRECTLY LOAD PRE-TRAINED MODEL FOR PREDICTION
# ENABLE THIS VARIABLE TO TEST NEW TEST SET WITHOUT TRAINING
LOAD_MODEL_FROM_FILE = False

# set automatic feature ranking and selection
AUTO_FEATURE_SELECTION=True
# set manually selected feature index list here
# check random forest setting when changing this variable
MANUAL_SELECTED_FEATURES = []

# The number of CPUs to use to do the computation. -1 means 'all CPUs'
NUM_CPU = -1
N_FOLD_VALIDATION = 5
TEST_SPLIT_PERCENT = 0.25
DNN_FEATURE_SIZE = 300


#####################################################


class ChaseClassifier(object):
    """
    """

    def __init__(self, task, identifier,
                 data_file,
                 feat_v: fv.FeatureVectorizer,
                 folder_sysout):
        self.raw_data = numpy.empty
        self.data_file = data_file
        self.identifier = identifier
        self.task_name = task
        self.feat_v = feat_v  # inclusive
        self.sys_out = folder_sysout  # exclusive 16
        self.feature_size = DNN_FEATURE_SIZE

    def load_data(self):
        self.raw_data = pd.read_csv(self.data_file, sep=',', encoding="utf-8")

    def scale_data(self, data):
        if SCALING_STRATEGY == SCALING_STRATEGY_MEAN_STD:
            data = util.feature_scaling_mean_std(data)
        elif SCALING_STRATEGY == SCALING_STRATEGY_MIN_MAX:
            data = util.feature_scaling_min_max(data)
        else:
            raise ArithmeticError("SCALING STRATEGY IS NOT SET CORRECTLY!")
        return data

    def training(self):
        tweets = self.raw_data.tweet
        tweets = [x for x in tweets if type(x) == str]
        print("FEATURE EXTRACTION AND VECTORIZATION FOR ALL data, insatance={}, {}"
              .format(len(tweets), datetime.datetime.now()))
        print("\tbegin feature extraction and vectorization...")
        tweets_cleaned = [tp.preprocess(x) for x in tweets]
        M = self.feat_v.transform_inputs(tweets, tweets_cleaned, self.sys_out, "na")
        #M = pd.DataFrame(M)

        if AUTO_FEATURE_SELECTION:
            select = SelectFromModel(LogisticRegression(class_weight='balanced',penalty="l1",C=0.01))
            M = select.fit_transform(M, self.raw_data['class'])

        # split the dataset into two parts, 0.75 for train and 0.25 for testing
        X_train_data, X_test_data, y_train, y_test = \
            train_test_split(M, self.raw_data['class'],
                             test_size=TEST_SPLIT_PERCENT,
                             random_state=42)
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)


        ######################### SGDClassifier #######################
        if WITH_SGD:
            cl.learn_generative(NUM_CPU, N_FOLD_VALIDATION, self.task_name, LOAD_MODEL_FROM_FILE, "sgd",
                                X_train_data, y_train,
                                X_test_data, y_test, self.identifier)

        ######################### Stochastic Logistic Regression#######################
        if WITH_SLR:
            cl.learn_generative(NUM_CPU, N_FOLD_VALIDATION, self.task_name, LOAD_MODEL_FROM_FILE, "lr",
                                X_train_data, y_train,
                                X_test_data, y_test, self.identifier)

        ######################### Random Forest Classifier #######################
        if WITH_RANDOM_FOREST:
            cl.learn_discriminative(NUM_CPU, N_FOLD_VALIDATION, self.task_name, LOAD_MODEL_FROM_FILE, "rf",
                                    X_train_data,
                                    y_train,
                                    X_test_data, y_test, self.identifier)

        ###################  liblinear SVM ##############################
        if WITH_LIBLINEAR_SVM:
            cl.learn_discriminative(NUM_CPU, N_FOLD_VALIDATION, self.task_name, LOAD_MODEL_FROM_FILE,
                                    "svm-l", X_train_data,
                                    y_train, X_test_data, y_test, self.identifier)

        ##################### RBF svm #####################
        if WITH_RBF_SVM:
            cl.learn_discriminative(NUM_CPU, N_FOLD_VALIDATION, self.task_name, LOAD_MODEL_FROM_FILE,
                                    "svm-rbf", X_train_data,
                                    y_train, X_test_data, y_test, self.identifier)

        ################# Artificial Neural Network #################
        if WITH_ANN:
            cl.learn_dnn(NUM_CPU, N_FOLD_VALIDATION, self.task_name, LOAD_MODEL_FROM_FILE, "ann",
                         self.feature_size,
                         X_train_data,
                         y_train, X_test_data, y_test, self.identifier)

        print("complete!")

    #todo: this is not completed. feature dimension must be the same as training data
    def testing(self):
        tweets = self.raw_data.tweet
        tweets = [x for x in tweets if type(x) == str]
        print("FEATURE EXTRACTION AND VECTORIZATION FOR Training data, insatance={}, {}"
              .format(len(tweets), datetime.datetime.now()))
        print("\tbegin feature extraction and vectorization...")
        tweets_cleaned = [tp.preprocess(x) for x in tweets]
        M_train = self.feat_v.transform_inputs(tweets, tweets_cleaned, self.sys_out)
        featurematrix = pd.DataFrame(M_train)
        labels = self.raw_data.hatespeech['class'].astype(int)

        print("Applying pre-trained models to tag data (i.e., testing) :: testing data size:", len(tweets))
        print("test with CPU cores: [%s]" % NUM_CPU)

        ######################### SGDClassifier #######################
        if WITH_SGD:
            ct.tag(NUM_CPU, "sgd", self.task_name, featurematrix)

        ######################### Stochastic Logistic Regression#######################
        if WITH_SLR:
            ct.tag(NUM_CPU, "lr", self.task_name, featurematrix)

        ######################### Random Forest Classifier #######################
        if WITH_RANDOM_FOREST:
            ct.tag(NUM_CPU, "rf", self.task_name, featurematrix)

        ###################  liblinear SVM ##############################
        if WITH_LIBLINEAR_SVM:
            ct.tag(NUM_CPU, "svm-l", self.task_name, featurematrix)
        ##################### RBF svm #####################
        if WITH_RBF_SVM:
            ct.tag(NUM_CPU, "svm-rbf", self.task_name, featurematrix)
        print("complete!")

    def saveOutput(self, prediction, model_name):
        filename = os.path.join(os.path.dirname(__file__), "prediction-%s-%s.csv" % (model_name, self.task_name))
        file = open(filename, "w")
        for entry in prediction:
            if (isinstance(entry, float)):
                file.write(str(entry) + "\n")
                # file.write("\n")
            else:
                if (entry[0] > entry[1]):
                    file.write("0\n")
                else:
                    file.write("1\n")
        file.close()


if __name__ == '__main__':
    settings = experiment_settings.create_settings()
    for ds in settings:
        print("\nSTARTING EXPERIMENT SETTING:" + '; '.join(map(str, ds)))
        classifier = ChaseClassifier(ds[0], ds[1], ds[2], ds[3], ds[4])
        classifier.load_data()

        # ============= random sampling =================================
        # print("training data size before resampling:", len(classifier.training_data))
        # X_resampled, y_resampled = classifier.under_sampling(classifier.training_data,                                                         classifier.training_label)
        # print("training data size after resampling:", len(X_resampled))
        # enable this line to visualise the data
        # classifier.training_data = X_resampled
        # classifier.training_label = y_resampled

        classifier.training()
        # classifier.testing()
