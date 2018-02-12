import csv


in_file="/home/zz/SCORES_w.csv"
out_file="/home/zz/SCORES_w_dm1.csv"
with open(in_file, newline='') as csvfile:
    csvr = csv.reader(csvfile, delimiter=',', quotechar='"')

    with open(out_file, 'w', newline='\n') as csvfile:
        csvw = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)

        delete=False
        for row in csvr:
            if len(row)==0:
                csvw.writerow([""])
                continue
            if row[0]==" N-FOLD AVERAGE :" or row[0]==" HELDOUT :":
                delete=True
                continue
            if delete:
                delete=False
                continue
            if row[0]==" for data from w :":
                continue

            csvw.writerow(row)

