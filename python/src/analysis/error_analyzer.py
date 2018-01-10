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

    for i in range(0, len(_all_predictions[0])):
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
            raw_data_row=X_train_data[i]
            msg=raw_data_row[7]
            label=raw_data_row[6]
            row=[label,msg]
            row=row+annotations
            writer.writerow(row)


gs_data_file = "/home/zqz/Work/chase/data/ml/ml/ws-exp/labeled_data_all.csv"
error_folder = "/home/zqz/Work/chase/output/error_analysis/ws-exp"
out_folder = "/home/zqz/Work/chase/output/error_analysis"

collect_wrong_predictoins(gs_data_file,error_folder,out_folder)

