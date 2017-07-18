import csv
import pickle

import datetime

from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import unique_labels

from ml import text_preprocess


def load_classifier_model(classifier_pickled=None):
    if classifier_pickled:
        with open(classifier_pickled, 'rb') as model:
            classifier = pickle.load(model)
        return classifier


def outputFalsePredictions(pred, truth, model_name, task, outfolder):
    subfolder = outfolder + "/errors"
    try:
        os.stat(subfolder)
    except:
        os.mkdir(subfolder)
    filename = os.path.join(subfolder, "errors-%s-%s.csv" % (model_name, task))
    file = open(filename, "w")
    for p, t in zip(pred, truth):
        if p == t:
            line = str(p) + ",ok\n"
            file.write(line)
        else:
            line = str(p) + ",wrong\n"
            file.write(line)
    file.close()


def prepare_score_string(p, r, f1, s, labels, target_names, digits):
    string = ",precision, recall, f1, support\n"
    for i, label in enumerate(labels):
        string = string + target_names[i] + ","
        for v in (p[i], r[i], f1[i]):
            string = string + "{0:0.{1}f}".format(v, digits) + ","
        string = string + "{0}".format(s[i]) + "\n"
        # values += ["{0}".format(s[i])]
        # report += fmt % tuple(values)

    # average
    string += "avg,"
    for v in (np.average(p),
              np.average(r),
              np.average(f1)):
        string += "{0:0.{1}f}".format(v, digits) + ","
    string += '{0}'.format(np.sum(s)) + "\n\n"
    return string


def save_scores(nfold_predictions, x_test, heldout_predictions, y_test, model_name, task_name,
                identifier, digits, outfolder):
    outputFalsePredictions(nfold_predictions, x_test, model_name, task_name, outfolder)
    subfolder = outfolder + "/scores"
    try:
        os.stat(subfolder)
    except:
        os.mkdir(subfolder)
    filename = os.path.join(subfolder, "scores-%s-%s.csv" % (model_name, task_name))
    file = open(filename, "a+")
    file.write(identifier)
    if nfold_predictions is not None:
        file.write("N-fold results:\n")
        labels = unique_labels(x_test, nfold_predictions)
        target_names = ['%s' % l for l in labels]
        p, r, f1, s = precision_recall_fscore_support(x_test, nfold_predictions,
                                                      labels=labels)
        line = prepare_score_string(p, r, f1, s, labels, target_names, digits)
        file.write(line)

    if (heldout_predictions is not None):
        file.write("Heldout results:\n")
        labels = unique_labels(y_test, heldout_predictions)
        target_names = ['%s' % l for l in labels]
        p, r, f1, s = precision_recall_fscore_support(y_test, heldout_predictions,
                                                      labels=labels)
        line = prepare_score_string(p, r, f1, s, labels, target_names, digits)
        file.write(line)
    file.close()


def index_max(values):
    return max(range(len(values)), key=values.__getitem__)


def save_classifier_model(model, outfile):
    if model:
        with open(outfile, 'wb') as model_file:
            pickle.dump(model, model_file)


def print_eval_report(best_params, cv_score, prediction_dev,
                      time_predict_dev,
                      time_train, y_test):
    print("CV score [%s]; best params: [%s]" %
          (cv_score, best_params))
    print("\nTraining time: %fs; "
          "Prediction time for 'dev': %fs;" %
          (time_train, time_predict_dev))
    print("\n %fs fold cross validation score:" % cv_score)
    print("\n test set result:")
    print("\n" + classification_report(y_test, prediction_dev))


def timestamped_print(msg):
    ts = str(datetime.datetime.now())
    print(ts + " :: " + msg)


def validate_training_set(training_set):
    """
    validate training data set (i.e., X) before scaling, PCA, etc.
    :param training_set: training set, test data
    :return:
    """
    # print("np any isnan(X): ", np.any(np.isnan(training_set)))
    # print("np all isfinite: ", np.all(np.isfinite(training_set)))
    # check any NaN row
    row_i = 0
    for i in training_set:
        row_i += 1
        if np.any(np.isnan(i)):
            print("ERROR: [", row_i, "] is nan: ")
            print(i)


def feature_scaling_mean_std(feature_set):
    scaler = StandardScaler(with_mean=True, with_std=True)
    return scaler.fit_transform(feature_set)


def feature_scaling_min_max(feature_set):
    """
    Input X must be non-negative for multinomial Naive Bayes model
    :param feature_set:
    :return:
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(feature_set)


def class_to_name(class_label):
    """
    This function can be used to map a numeric
    feature name to a particular class.
    """
    if class_label == 0:
        return "Hate speech"
    elif class_label == 1:
        return "Offensive language"
    elif class_label == 2:
        return "Neither"
    else:
        return "No label"


def name_to_class(class_label):
    # U= unknown, R = Religion, E = Ethnicity, S = Sexuality, Y = yes blank = no, x = don't use
    if (class_label == "r") or (class_label == "e") or (class_label == "s") or (class_label == "y"):
        return "0"  # Hate speech
    elif class_label == "":
        return "2"  # neither
    else:
        return "x"  # dont use


def output_data_splits(data_file, out_folder):
    raw_data = pd.read_csv(data_file, sep=',', encoding="utf-8")
    X_train_data, X_test_data, y_train, y_test = \
        train_test_split(raw_data, raw_data['class'],
                         test_size=0.2,
                         random_state=42)
    X_train_data.to_csv(out_folder + "/split_train.csv", sep=',', encoding='utf-8')
    X_test_data.to_csv(out_folder + "/split_test.csv", sep=',', encoding='utf-8')


def save_selected_features(finalFeatureIndices, featureTypes, file):
    with open(file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)

        start = 0
        featureTypesAsDict = {}
        # for item in featureTypes:
        #     if isinstance(item, tuple):
        #         featureTypesAsDict[item[0]]=item[1]
        #     elif isinstance(item, list):
        #         i = iter(item)


        for ft_key, ft_value in featureTypes.items():
            if isinstance(ft_value[1], dict):
                feature_lookup = {v: k for k, v in ft_value[1].items()}
            else:
                feature_lookup = {v: k for v, k in enumerate(ft_value[1])}
            max = start + len(feature_lookup)
            for i in finalFeatureIndices:
                if i < start:
                    continue
                if i < max:
                    feature = feature_lookup[i - start]
                    writer.writerow([i, ft_key, feature])
            start = max

    return None


def saveOutput(prediction, model_name, task, outfolder):
    filename = os.path.join(outfolder, "prediction-%s-%s.csv" % (model_name, task))
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


def feature_scale(option, M):
    # if self.feature_selection:
    #     print("FEATURE SELECTION BEGINS, {}".format(datetime.datetime.now()))
    #     select = SelectFromModel(LogisticRegression(class_weight='balanced',penalty="l1",C=0.01))
    #     M = select.fit_transform(M, self.raw_data['class'])
    #     print("REDUCED FEATURE MATRIX dimensions={}".format(M.shape))
    # if not self.feature_selection:
    print("APPLYING FEATURE SCALING: [%s]" % option)
    if option == 0:  # mean std
        M = feature_scaling_mean_std(M)
    elif option == 1:
        M = feature_scaling_min_max(M)
    else:
        pass

    # print("FEATURE SELECTION BEGINS, {}".format(datetime.datetime.now()))
    # select = SelectFromModel(LogisticRegression(class_weight='balanced',penalty="l1",C=0.01))
    # M = select.fit_transform(M, self.raw_data['class'])
    # print("REDUCED FEATURE MATRIX dimensions={}".format(M.shape))

    return M


def feature_extraction(data_column, feat_vectorizer, sysout):
    tweets = data_column
    tweets = [x for x in tweets if type(x) == str]
    print("FEATURE EXTRACTION AND VECTORIZATION FOR ALL data, insatance={}, {}"
          .format(len(tweets), datetime.datetime.now()))
    print("\tbegin feature extraction and vectorization...")
    tweets_cleaned = [text_preprocess.preprocess_clean(x,1,1) for x in tweets]
    M = feat_vectorizer.transform_inputs(tweets, tweets_cleaned, sysout, "na")
    print("FEATURE MATRIX dimensions={}".format(M[0].shape))
    return M


def read_preselected_features(only_intersection, *files):
    file_with_features = []
    for file in files:
        feature_with_values = {}
        with open(file, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for row in csvreader:
                ft = row[1]
                value = row[2]
                if (ft in feature_with_values.keys()):
                    feature_with_values[ft].append(value)
                else:
                    values=[]
                    values.append(value)
                    feature_with_values[ft]=values
            file_with_features.append(feature_with_values)

    all_fts=set()
    all_fts.update(file_with_features[0].keys())
    for i in range(1, len(file_with_features)):
        all_fts=set.intersection(all_fts, file_with_features[i].keys())


    selected_features={}
    for ft in all_fts:
        selected=[]
        for file_features in file_with_features:
            values=file_features[ft]
            selected.append(set(values))

        if only_intersection:
            selected_features[ft]=set.intersection(*selected)
        else:
            selected_features[ft]=set.union(*selected)

    return selected_features


# read_preselected_features(True,"/home/zqz/Work/chase/output/models/td-tdf/svml-td-tdf-kb.m.features.csv",
#                           "/home/zqz/Work/chase/output/models/td-tdf/svml-td-tdf-sfm.m.features.csv",
#                           "/home/zqz/Work/chase/output/models/td-tdf/svml-td-tdf-rfe.m.features.csv")
