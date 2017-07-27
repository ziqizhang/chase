import csv

import logging
import numpy
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
import os
from ml import util
LOG_DIR = os.getcwd() + "/logs"
logger = logging.getLogger(__name__)
logging.basicConfig(filename=LOG_DIR + '/classifier.log', level=logging.INFO, filemode='w')

def create_classifier(model, sysout, task, cpus, input_dim):
    classifier=None
    model_file=None
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

    return classifier, model_file


def transform_test_features(data, feature_vectorizer,
                            training_feature_save,
                            sys_out,
                            scaling_option):
    # test data must be represented in a feature matrix of the same dimension of the training data feature matrix
    # step 1: reconstruct empty feature matrix using the vocabularies seen at training time
    logger.info("\n\nEXTRACTING TEST DATA FEATURS...")
    meta_TEST = util.feature_extraction(data, feature_vectorizer, sys_out, logger)

    logger.info("\nFEATURE SELECTION ON TEST DATA...")
    train_features = create_training_features(training_feature_save)
    # step 2: create test data features
    M_features_by_type = meta_TEST[1]
    # step 3: map test data features to training data features and populate the empty feature matrix
    featurematrix = map_to_trainingfeatures(train_features, M_features_by_type)
    featurematrix = util.feature_scale(scaling_option,featurematrix)

    return featurematrix


def create_training_features(saved_training_feature_vocab):
    rs={}
    with open(saved_training_feature_vocab, newline='',encoding='utf-8') as csvfile:
        reader= csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            featureType=row[1]
            featureValue=row[2]

            if featureType in rs.keys():
                rs[featureType].append(featureValue)
            else:
                featureValues=[]
                featureValues.append(featureValue)
                rs[featureType]=featureValues
    #
    # for ft in list_of_expected_feature_types:
    #     file = saved_training_feature_vocab + "/" + ft + ".pk"
    #     vocab = pickle.load(open(file, "rb"))
    #     rs[ft] = vocab
    return rs


'''keep_features: a dictionary where key is type of feature, value can be a list of feature values,
or a dictionary that contains index-feature value pairs.
from_features: this is the object created by feature_vectorizer. it contains a dictionary where
key is the feature type, value is a mxn matrix, m is #instances in data, n is feature values for that type
filter: if we only want to keep certain rows from the mxn matrix from *from_features*, the filter
should be passed as list of the indices in that matrix'''
def map_to_trainingfeatures(keep_features: {}, from_features: {},
                            filter=None):
    if filter is None:
        num_instances = len(next(iter(from_features.values()))[0])
    else:
        num_instances=len(filter)

    new_features = []

    # for each feature type in train data features
    for t_key, t_value in keep_features.items():
        t_value_=t_value
        if isinstance(t_value_, set):
            t_value_=list(t_value)

        logger.info("\t mapping feature type={}, features={}".format(t_key, len(t_value_)))
        features = numpy.zeros((num_instances, len(t_value_)))
        # if feature type exist in test data features
        if t_key in from_features:
            # for each feature in test data feature of this type
            fromdata_feature = from_features[t_key]
            fromdata_feature_vocab = fromdata_feature[1]
            fromdata_feature_value = fromdata_feature[0]
            # for each data instance
            from_features_row_index = 0
            row_index=0
            for row in fromdata_feature_value:
                if filter is not None and from_features_row_index not in filter:
                    from_features_row_index += 1
                    continue

                new_row = numpy.zeros(len(t_value_))
                for vocab, value in zip(fromdata_feature_vocab, row):  # feature-value pair for each instance in test
                    # check if that feature exists in train data features of that type
                    if vocab in t_value_:
                        # if so, se corresponding feature value in the new feature vector
                        if isinstance(t_value_, dict):
                            vocab_index = t_value_[vocab]
                        else:
                            vocab_index = t_value_.index(vocab)
                        new_row[vocab_index] = value
                features[row_index] = new_row
                if(from_features_row_index%100==0):
                    logger.info("(progress: {})".format(from_features_row_index))
                from_features_row_index += 1
                row_index+=1
        else:
            features = numpy.zeros(num_instances, len(t_value_))
        new_features.append(features)

    M = numpy.concatenate(new_features, axis=1)
    return M
