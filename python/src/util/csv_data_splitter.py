import csv


in_file="/home/zz/Work/chase/data/ml/ml/rm/labeled_data_all.csv"
out_file="/home/zz/Work/chase/data/ml/ml/rm/labeled_data_tweets_only.csv"
with open(in_file, newline='') as csvfile:
    csvr = csv.reader(csvfile, delimiter=',', quotechar='"')

    with open(out_file, 'w', newline='\n') as csvfile:
        csvw = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)

        delete=False
        for row in csvr:
            csvw.writerow([row[6],row[7]])

