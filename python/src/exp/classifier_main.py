#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import datetime
import pickle

from exp import experiment_settings
from ml.vectorizer import feature_vectorizer as fv
from ml import classifier_train as cl
from ml import classifier_tag as ct
import numpy
from sklearn.model_selection import train_test_split
import pandas as pd
from ml import util
from ml import text_preprocess as tp
from ml import feature_extractor as fe

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
SCALING_STRATEGY = SCALING_STRATEGY_MEAN_STD

# DIRECTLY LOAD PRE-TRAINED MODEL FOR PREDICTION
# ENABLE THIS VARIABLE TO TEST NEW TEST SET WITHOUT TRAINING
LOAD_MODEL_FROM_FILE = False
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

    def training(self):
        meta_M=self.feature_extraction()
        M=meta_M[0]
        M=self.feature_scale(M)

        # split the dataset into two parts, 0.75 for train and 0.25 for testing
        X_train_data, X_test_data, y_train, y_test = \
            train_test_split(M, self.raw_data['class'],
                             test_size=TEST_SPLIT_PERCENT,
                             random_state=42)
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


    def testing(self, expected_feature_types, training_feature_vocab_folder, sys_out):
        #test data must be represented in a feature matrix of the same dimension of the training data feature matrix
        #step 1: reconstruct empty feature matrix using the vocabularies seen at training time
        train_features=self.create_training_features(expected_feature_types, training_feature_vocab_folder)
        #step 2: create test data features
        M=self.feature_extraction()
        M_features_by_type=M[1]
        #step 3: map test data features to training data features and populate the empty feature matrix
        featurematrix=self.map_to_trainingfeatures(train_features, M_features_by_type)

        featurematrix=self.feature_scale(featurematrix)

        print("Applying pre-trained models to tag data (i.e., testing) :: testing data size:", len(self.raw_data))
        print("test with CPU cores: [%s]" % NUM_CPU)

        ######################### SGDClassifier #######################
        if WITH_SGD:
            ct.tag(NUM_CPU, "sgd", self.task_name, featurematrix, sys_out)

        ######################### Stochastic Logistic Regression#######################
        if WITH_SLR:
            ct.tag(NUM_CPU, "lr", self.task_name, featurematrix,sys_out)

        ######################### Random Forest Classifier #######################
        if WITH_RANDOM_FOREST:
            ct.tag(NUM_CPU, "rf", self.task_name, featurematrix,sys_out)

        ###################  liblinear SVM ##############################
        if WITH_LIBLINEAR_SVM:
            ct.tag(NUM_CPU, "svm-l", self.task_name, featurematrix,sys_out)
        ##################### RBF svm #####################
        if WITH_RBF_SVM:
            ct.tag(NUM_CPU, "svm-rbf", self.task_name, featurematrix,sys_out)
        print("complete at {}".format(datetime.datetime.now()))


    def feature_extraction(self):
        tweets = self.raw_data.tweet
        tweets = [x for x in tweets if type(x) == str]
        print("FEATURE EXTRACTION AND VECTORIZATION FOR ALL data, insatance={}, {}"
              .format(len(tweets), datetime.datetime.now()))
        print("\tbegin feature extraction and vectorization...")
        tweets_cleaned = [tp.preprocess(x) for x in tweets]
        M = self.feat_v.transform_inputs(tweets, tweets_cleaned, self.sys_out, "na")
        print("FEATURE MATRIX dimensions={}".format(M[0].shape))
        return M


    def create_training_features(self, list_of_expected_feature_types, saved_training_feature_vocab):
        rs={}
        for ft in list_of_expected_feature_types:
            file=saved_training_feature_vocab+"/"+ft+".pk"
            vocab=pickle.load(open(file, "rb" ))
            rs[ft]=vocab
        return rs


    def map_to_trainingfeatures(self, training_features:{}, testdata_features:{}):
        num_instances=len(next (iter (testdata_features.values()))[0])
        new_features=[]

        #for each feature type in train data features
        for t_key, t_value in training_features.items():
            print("\t mapping feature type={}, features={}".format(t_key, len(t_value)))
            features=numpy.zeros((num_instances, len(t_value)))
            #if feature type exist in test data features
            if t_key in testdata_features:
                #for each feature in test data feature of this type
                testdata_feature = testdata_features[t_key]
                testdata_feature_vocab=testdata_feature[1]
                testdata_feature_value=testdata_feature[0]
                #for each data instance
                row_index=0
                for row in testdata_feature_value:
                    new_row = numpy.zeros(len(t_value))
                    for vocab, value in zip (testdata_feature_vocab, row): #feature-value pair for each instance in test
                        #check if that feature exists in train data features of that type
                        if vocab in t_value:
                            #if so, se corresponding feature value in the new feature vector
                            if isinstance(t_value, dict):
                                vocab_index=t_value[vocab]
                            else:
                                vocab_index=t_value.index(vocab)
                            new_row[vocab_index]=value
                    features[row_index]= new_row
                    row_index+=1
            else:
                features=numpy.zeros(num_instances, len(t_value))
            new_features.append(features)

        M = numpy.concatenate(new_features, axis=1)
        return M


    def feature_scale(self, M):
        # if self.feature_selection:
        #     print("FEATURE SELECTION BEGINS, {}".format(datetime.datetime.now()))
        #     select = SelectFromModel(LogisticRegression(class_weight='balanced',penalty="l1",C=0.01))
        #     M = select.fit_transform(M, self.raw_data['class'])
        #     print("REDUCED FEATURE MATRIX dimensions={}".format(M.shape))
        #if not self.feature_selection:
        print("APPLYING FEATURE SCALING: [%s]" % SCALING_STRATEGY)
        if SCALING_STRATEGY == SCALING_STRATEGY_MEAN_STD:
            M = util.feature_scaling_mean_std(M)
        elif SCALING_STRATEGY == SCALING_STRATEGY_MIN_MAX:
            M = util.feature_scaling_min_max(M)
        else:
            pass
        return M

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
    settings = experiment_settings.create_settings(sys.argv[1], sys.argv[2])
    for ds in settings:
        print("##########\nSTARTING EXPERIMENT SETTING:" + '; '.join(map(str, ds)))
        classifier = ChaseClassifier(ds[0], ds[1], ds[2], ds[3], ds[4],ds[5],ds[6], ds[7],
                                     ds[8],ds[9])
        classifier.load_data()

        # ============= random sampling =================================
        # print("training data size before resampling:", len(classifier.training_data))
        # X_resampled, y_resampled = classifier.under_sampling(classifier.training_data,                                                         classifier.training_label)
        # print("training data size after resampling:", len(X_resampled))
        # enable this line to visualise the data
        # classifier.training_data = X_resampled
        # classifier.training_label = y_resampled

        classifier.training()
        # classifier.testing([fe.NGRAM_FEATURES_VOCAB, fe.NGRAM_POS_FEATURES_VOCAB, fe.TWEET_TD_OTHER_FEATURES_VOCAB],
        #                    sys.argv[1]+"/features",
        #                    sys.argv[1])
