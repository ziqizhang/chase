import csv
import os

import pandas as pd
# given a gs_data file, find the corresponding splits used in experiment (75:25, the 25 part).
# given error files by each model, find the errors made by ALL models.
# output the message, the class, to outfolder
from sklearn.cross_validation import train_test_split

def collect_wrong_predictoins(gs_data_file,error_folder,out_folder):
    raw_data = pd.read_csv(gs_data_file, sep=',', encoding="utf-8")
    X_train_data, X_test_data, y_train, y_test = \
        train_test_split(raw_data, raw_data['class'],
                         test_size=0.25,
                         random_state=42)
    X_train_data=X_train_data.as_matrix()

    _all_predictions=[]
    for file in os.listdir(error_folder):
        file_path = error_folder + "/" + file

        predictions = []
        with open(file_path, encoding="utf8", newline='\n') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            count=0
            for row in csvreader:
                if count==0:
                    count+=1
                    continue
                predictions.append(row)
            _all_predictions.append(predictions)

    filename=out_folder+"/"+error_folder[error_folder.rindex("/")+1:]+".csv"
    writer = csv.writer(open(filename, "w",encoding="utf8"))


    class_distr={}
    error_distr={}
    for i in range(0, len(_all_predictions[0])):
        raw_data_row=X_train_data[i]
        gs_label=raw_data_row[6]

        if gs_label in class_distr.keys():
            class_distr[gs_label]+=1
        else:
            class_distr[gs_label]=1

        select=True
        annotations=[]
        for pred in _all_predictions:
            row=pred[i]
            if row[1]!="wrong":
                select=False
                break
            else:
                annotations.append(row[0])

        if select:
            if gs_label in error_distr.keys():
                error_distr[gs_label]+=1
            else:
                error_distr[gs_label]=1

            msg=raw_data_row[7]
            row=[gs_label,msg]
            row=row+annotations
            writer.writerow(row)

    for k, v in class_distr.items():
        print(str(k)+","+str(v))

    for k, v in error_distr.items():
        print(str(k)+","+str(v))

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

gs_data_file = "/home/zqz/Work/chase/data/ml/ml/dt/labeled_data_all_2.csv"
error_folder = "/home/zqz/Work/chase/output/error_analysis/dt"
out_folder = "/home/zqz/Work/chase/output/error_analysis"

collect_wrong_predictoins(gs_data_file,error_folder,out_folder)

