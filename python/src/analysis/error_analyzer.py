import csv
import os

import pandas as pd
# given a gs_data file, find the corresponding splits used in experiment (75:25, the 25 part).
# given error files by each model, find the errors made by ALL models.
# output the message, the class, to outfolder
from sklearn.cross_validation import train_test_split


def collect_wrong_predictoins(gs_data_file, error_folder, out_folder,
                              tweet_col, label_col,rowid_col=None):
    raw_data = pd.read_csv(gs_data_file, sep=',', encoding="utf-8")
    X_train_data, X_test_data, y_train, y_test = \
        train_test_split(raw_data, raw_data['class'],
                         test_size=0.25,
                         random_state=42)
    X_train_data = X_train_data.as_matrix()

    _all_wrong_predictions = []
    for file in os.listdir(error_folder):
        file_path = error_folder + "/" + file

        predictions = []
        with open(file_path, encoding="utf8", newline='\n') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            count = 0
            for row in csvreader:
                if count == 0:
                    count += 1
                    continue
                if row[2] == 'wrong':
                    predictions.append(row[0])
            _all_wrong_predictions.append(predictions)

    # find overlap
    s = set(_all_wrong_predictions[0])
    for i in range(1, len(_all_wrong_predictions)):
        s = s & set(_all_wrong_predictions[i])

    filename = out_folder + "/" + error_folder[error_folder.rindex("/") + 1:] + ".csv"
    writer = csv.writer(open(filename, "w", encoding="utf8"))

    count=0
    if rowid_col is not None:
        for err in s:
            count+=1
            if count==8:
                print()
            r = list(raw_data.loc[raw_data[rowid_col] == int(err)].iloc[0])
            line = [err, r[label_col], r[tweet_col]]
            writer.writerow(line)
            print(count)
    else:
        for err in s:
            r = list(raw_data.iloc[int(err)])
            line = [err, r[label_col], r[tweet_col]]
            writer.writerow(line)


# gs_data_file = "/home/zqz/Work/chase/data/ml/ml/ws-exp/labeled_data_all.csv"
# error_folder = "/home/zqz/Work/chase/output/error_analysis/ws-exp"
# out_folder = "/home/zqz/Work/chase/output/error_analysis"


# gs_data_file = "/home/zqz/Work/chase/data/ml/ml/ws-amt/labeled_data_all.csv"
# error_folder = "/home/zqz/Work/chase/output/error_analysis/ws-amt"
# out_folder = "/home/zqz/Work/chase/output/error_analysis"

# gs_data_file = "/home/zqz/Work/chase/data/ml/ml/ws-gb/labeled_data_all.csv"
# error_folder = "/home/zqz/Work/chase/output/error_analysis/ws-gb"
# out_folder = "/home/zqz/Work/chase/output/error_analysis"

# gs_data_file = "/home/zqz/Work/chase/data/ml/ml/w+ws/labeled_data_all.csv"
# error_folder = "/home/zqz/Work/chase/output/error_analysis/wws"
# out_folder = "/home/zqz/Work/chase/output/error_analysis"

# gs_data_file = "/home/zqz/Work/chase/data/ml/ml/rm/labeled_data_all.csv"
# error_folder = "/home/zqz/Work/chase/output/error_analysis/rm"
# out_folder = "/home/zqz/Work/chase/output/error_analysis"

# gs_data_file = "/home/zz/Work/chase/data/ml/ml/rm/labeled_data_all.csvc.csv"
# error_folder = "/home/zz/Work/chase/output/errors/rm"
# out_folder = "/home/zz/Work/chase/output/error_analysis"
gs_data_file = "/home/zz/Work/chase/data/ml/ml/w+ws/labeled_data_all.csvc.csv"
error_folder = "/home/zz/Work/chase/output/errors/wws"
out_folder = "/home/zz/Work/chase/output/error_analysis"

#rm- 7, 6
#wws- 7, 6
#ws-exp- 7, 6
#dt- Unamed: 0, 6, 5
collect_wrong_predictoins(gs_data_file, error_folder, out_folder,
                          7,6)
