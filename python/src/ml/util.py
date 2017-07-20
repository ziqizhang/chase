import csv
import pickle

import datetime
import random
import pandas
from sklearn.cross_validation import train_test_split
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


def save_scores(nfold_predictions, nfold_truth,
                heldout_predictions, heldout_truth,
                model_name, task_name,
                identifier, digits, outfolder,
                instance_data_source_tags: pandas.Series = None, accepted_ds_tags: list = None):
    outputFalsePredictions(nfold_predictions, nfold_truth, model_name, task_name, outfolder)
    subfolder = outfolder + "/scores"
    try:
        os.stat(subfolder)
    except:
        os.mkdir(subfolder)
    filename = os.path.join(subfolder, "SCORES__%s_%s.csv" % (model_name, task_name))
    writer = open(filename, "a+")
    writer.write(identifier)
    if nfold_predictions is not None:
        writer.write(" N-FOLD AVERAGE :\n")
        write_scores(nfold_predictions, nfold_truth, digits, writer, instance_data_source_tags, accepted_ds_tags)

    if (heldout_predictions is not None):
        writer.write(" HELDOUT :\n")
        write_scores(heldout_predictions, heldout_truth, digits, writer, instance_data_source_tags, accepted_ds_tags)

    writer.close()


def write_scores(predictoins, truth: pandas.Series, digits, writer, instance_dst_column=None,
                 accepted_ds_tags=None):
    labels = unique_labels(truth, predictoins)
    target_names = ['%s' % l for l in labels]
    p, r, f1, s = precision_recall_fscore_support(truth, predictoins,
                                                  labels=labels)
    line = prepare_score_string(p, r, f1, s, labels, target_names, digits)
    writer.write(line)

    if accepted_ds_tags is not None:
        for dstag in accepted_ds_tags:
            writer.write("\n for data from {} :\n".format(dstag))
            subset_pred = []
            subset_truth = []
            for index, label in zip(truth.index, predictoins):
                if instance_dst_column[index] == dstag:
                    subset_pred.append(label)
            for index, label in zip(truth.index, truth):
                if instance_dst_column[index] == dstag:
                    subset_truth.append(label)
            subset_labels = unique_labels(subset_truth, subset_pred)
            target_names = ['%s' % l for l in labels]
            p, r, f1, s = precision_recall_fscore_support(subset_truth, subset_pred,
                                                          labels=subset_labels)
            line = prepare_score_string(p, r, f1, s, subset_labels, target_names, digits)
            writer.write(line)


def index_max(values):
    return max(range(len(values)), key=values.__getitem__)


def save_classifier_model(model, outfile):
    if model:
        with open(outfile, 'wb') as model_file:
            pickle.dump(model, model_file)


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
    print("feature scaling, first perform sanity check...")
    if M.isnull().values.any():
        print("input matrix has NaN values, replace with 0")
        M.fillna(0)

    # if self.feature_selection:
    #     print("FEATURE SELECTION BEGINS, {}".format(datetime.datetime.now()))
    #     select = SelectFromModel(LogisticRegression(class_weight='balanced',penalty="l1",C=0.01))
    #     M = select.fit_transform(M, self.raw_data['class'])
    #     print("REDUCED FEATURE MATRIX dimensions={}".format(M.shape))
    # if not self.feature_selection:
    # logger.logger.info("APPLYING FEATURE SCALING: [%s]" % option)
    if option == 0:  # mean std
        M = feature_scaling_mean_std(M)
        if np.isnan(M).any():
            print("scaled matrix has NaN values, replace with 0")
        return np.nan_to_num(M)
    elif option == 1:
        M = feature_scaling_min_max(M)
        if np.isnan(M).any():
            print("scaled matrix has NaN values, replace with 0")
        return np.nan_to_num(M)
    else:
        pass

    # print("FEATURE SELECTION BEGINS, {}".format(datetime.datetime.now()))
    # select = SelectFromModel(LogisticRegression(class_weight='balanced',penalty="l1",C=0.01))
    # M = select.fit_transform(M, self.raw_data['class'])
    # print("REDUCED FEATURE MATRIX dimensions={}".format(M.shape))
    print("feature scaling done")
    return M


def feature_extraction(data_column, feat_vectorizer, sysout, logger):
    tweets = data_column
    tweets = [x for x in tweets if type(x) == str]
    logger.info("FEATURE EXTRACTION AND VECTORIZATION FOR ALL data, insatance={}, {}"
                .format(len(tweets), datetime.datetime.now()))
    logger.info("\tbegin feature extraction and vectorization...")
    tweets_cleaned = [text_preprocess.preprocess_clean(x, 1, 1) for x in tweets]
    M = feat_vectorizer.transform_inputs(tweets, tweets_cleaned, sysout, "na")
    logger.info("FEATURE MATRIX dimensions={}".format(M[0].shape))
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
                    values = []
                    values.append(value)
                    feature_with_values[ft] = values
            file_with_features.append(feature_with_values)

    all_fts = set()
    all_fts.update(file_with_features[0].keys())
    for i in range(1, len(file_with_features)):
        all_fts = set.intersection(all_fts, file_with_features[i].keys())

    selected_features = {}
    for ft in all_fts:
        selected = []
        for file_features in file_with_features:
            values = file_features[ft]
            selected.append(set(values))

        if only_intersection:
            selected_features[ft] = set.intersection(*selected)
        else:
            selected_features[ft] = set.union(*selected)

    return selected_features


def tag_source_file(csv_tdc_a, out_file):
    with open(out_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)

        with open(csv_tdc_a, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            count = 0
            for row in csvreader:
                if count == 0:
                    row.insert(0, "")
                    writer.writerow(row)
                    count += 1
                    continue

                if (len(row) > 7):
                    tweet_id = row[7]
                else:
                    row.insert(0, "td")
                    writer.writerow(row)
                    continue

                try:
                    float(tweet_id)
                except ValueError:
                    if len(row) > 8:
                        tweet_id = row[8]
                    else:
                        tweet_id = ""

                if len(tweet_id) == 0:
                    row.insert(0, "td")
                else:
                    row.insert(0, "c")
                writer.writerow(row)


def balanced_tdc_mixed(td_2_c_ratio, in_csv, out_csv):
    random.sample([1, 2, 3, 4, 5], 3)
    header = None
    c_rows = []
    td_rows = []
    with open(in_csv, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        count = 0
        for row in csvreader:
            if count == 0:
                header = row
                count += 1
                continue

            if row[0] == 'c':
                c_rows.append(row)
            else:
                td_rows.append(row)

    sample_size = int(td_2_c_ratio * len(c_rows))
    td_rows = random.sample(td_rows, sample_size)
    with open(out_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)
        for row in c_rows:
            writer.writerow(row)
        for row in td_rows:
            writer.writerow(row)


def separate_tdc(in_csv, out_csv, tag):
    with open(out_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)

        with open(in_csv, newline='', encoding='utf-8') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            count = 0
            for row in csvreader:
                if count == 0:
                    writer.writerow(row)
                    count += 1
                    continue

                if row[0] == tag:
                    writer.writerow(row)
                else:
                    continue

# separate_tdc("/home/zqz/Work/chase/data/ml/tdc-a/mixed_all.csv",
#              "/home/zqz/Work/chase/data/ml/c/labeled_data_all.csv", "c")

# tag_source_file("/home/zqz/Work/chase/data/ml/tdc-a/mixed_all.csv",
#                 "/home/zqz/Work/chase/data/ml/tdc-a/mixed_all_revised")

# balanced_tdc_mixed(1.1, "/home/zqz/Work/chase/data/ml/tdc-a/mixed_all.csv",
#                    "/home/zqz/Work/chase/data/ml/tdc-b/mixed_balance.csv")

# read_preselected_features(True,"/home/zqz/Work/chase/output/models/td-tdf/svml-td-tdf-kb.m.features.csv",
#                           "/home/zqz/Work/chase/output/models/td-tdf/svml-td-tdf-sfm.m.features.csv",
#                           "/home/zqz/Work/chase/output/models/td-tdf/svml-td-tdf-rfe.m.features.csv")
