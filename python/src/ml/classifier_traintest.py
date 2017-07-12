import pickle

import numpy
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import svm

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

from ml import classifier_gridsearch
from ml import util


def create_classifier(model, sysout, task, cpus, input_dim):
    subfolder = sysout + "/models"
    if (model == "rf"):
        classifier = RandomForestClassifier(n_estimators=20, n_jobs=cpus)
        model_file = subfolder + "/random-forest_classifier-%s.m" % task
    if (model == "svm-l"):
        classifier = svm.LinearSVC()
        model_file = subfolder + "/liblinear-svm-linear-%s.m" % task

    if (model == "svm-rbf"):
        classifier = svm.SVC()
        model_file = subfolder + "/liblinear-svm-rbf-%s.m" % task

    if (model == "sgd"):
        classifier = SGDClassifier(loss='log', penalty='l2', n_jobs=cpus)
        model_file = subfolder + "/sgd-classifier-%s.m" % task
    if (model == "lr"):
        classifier = LogisticRegression(random_state=111)
        model_file = subfolder + "/stochasticLR-%s.m" % task
    else:
        classifier = KerasClassifier(build_fn=classifier_gridsearch.create_model(input_dim), verbose=0)
        model_file = subfolder + "/ann-%s.m" % task

    return classifier, model_file


def transform_test_features(raw_test_data, feature_vectorizer,
                            expected_feature_types, training_feature_vocab_folder,
                            sys_out, feature_selected: bool,
                            scaling_option):
    # test data must be represented in a feature matrix of the same dimension of the training data feature matrix
    # step 1: reconstruct empty feature matrix using the vocabularies seen at training time
    meta_TEST = util.feature_extraction(raw_test_data.tweet, feature_vectorizer, sys_out)
    X_test = meta_TEST[0]
    if not feature_selected:
        return X_test

    train_features = create_training_features(expected_feature_types, training_feature_vocab_folder)
    # step 2: create test data features
    M_features_by_type = meta_TEST[1]
    # step 3: map test data features to training data features and populate the empty feature matrix
    featurematrix = map_to_trainingfeatures(train_features, M_features_by_type)
    featurematrix = util.feature_scale(featurematrix, scaling_option)

    return featurematrix


def create_training_features(list_of_expected_feature_types, saved_training_feature_vocab):
    rs = {}
    for ft in list_of_expected_feature_types:
        file = saved_training_feature_vocab + "/" + ft + ".pk"
        vocab = pickle.load(open(file, "rb"))
        rs[ft] = vocab
    return rs


def map_to_trainingfeatures(training_features: {}, testdata_features: {}):
    num_instances = len(next(iter(testdata_features.values()))[0])
    new_features = []

    # for each feature type in train data features
    for t_key, t_value in training_features.items():
        print("\t mapping feature type={}, features={}".format(t_key, len(t_value)))
        features = numpy.zeros((num_instances, len(t_value)))
        # if feature type exist in test data features
        if t_key in testdata_features:
            # for each feature in test data feature of this type
            testdata_feature = testdata_features[t_key]
            testdata_feature_vocab = testdata_feature[1]
            testdata_feature_value = testdata_feature[0]
            # for each data instance
            row_index = 0
            for row in testdata_feature_value:
                new_row = numpy.zeros(len(t_value))
                for vocab, value in zip(testdata_feature_vocab, row):  # feature-value pair for each instance in test
                    # check if that feature exists in train data features of that type
                    if vocab in t_value:
                        # if so, se corresponding feature value in the new feature vector
                        if isinstance(t_value, dict):
                            vocab_index = t_value[vocab]
                        else:
                            vocab_index = t_value.index(vocab)
                        new_row[vocab_index] = value
                features[row_index] = new_row
                row_index += 1
        else:
            features = numpy.zeros(num_instances, len(t_value))
        new_features.append(features)

    M = numpy.concatenate(new_features, axis=1)
    return M
