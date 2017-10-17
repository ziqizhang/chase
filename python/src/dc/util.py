import csv
import os
import pandas as pd


def merge_annotations(in_folder, out_file):
    tag_lookup={}
    id_lookup={}
    for file in sorted(os.listdir(in_folder)):
        print(file)
        with open(in_folder+"/"+file, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for row in reader:
                tag=row[0]
                id=row[2]
                ignore=False
                try:
                    val = float(id)
                except ValueError:
                    ignore=True
                    pass

                if ignore:
                    continue
                content=row[1]
                tag_lookup[content]=tag
                id_lookup[content]=id

    with open(out_file, 'w', newline='',encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for key, value in id_lookup.items():
            tag=tag_lookup[key]
            writer.writerow([tag,key, value])



def merge_waseem_datasets(in_large_dataset, in_small_dataset, out_file):
    with open(out_file, 'w', newline='',encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["ds","","count","hate_speech","offensive_language,","neither","class","tweet"])
        all_tweet_ids=set()
        data=pd.read_csv(in_large_dataset, sep=',', encoding="utf-8")
        index=0
        for row in data.itertuples():
            if index<1:
                index+=1
                continue

            tweetid=row[2]
            all_tweet_ids.add(tweetid)

            writer.writerow(row[1:])
            index+=1
            if index%100==0:
                print(index)

        data=pd.read_csv(in_small_dataset, sep=',', encoding="utf-8")
        index=0
        for row in data.itertuples():
            if index<1:
                index+=1
                continue
            tweetid=row[2]
            if tweetid in all_tweet_ids:
                continue

            writer.writerow(row[1:])
            index+=1
            if index%100==0:
                print(index)


def anonymize_dataset(in_file, out_file):
    with open(out_file, 'w', newline='',encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["id","class"])

        data=pd.read_csv(in_file, sep=',', encoding="utf-8")
        index=0
        for row in data.itertuples():
            if index<1:
                index+=1
                continue

            tweetid=row[2]
            clazz=row[7]

            writer.writerow([tweetid,clazz])
            index+=1
            if index%100==0:
                print(index)

anonymize_dataset("/home/zqz/Work/chase/data/ml/public/w+ws/labeled_data_all.csv",
                  "/home/zqz/Work/chase/data/ml/public/w+ws/labeled_data.csv")


# merge_waseem_datasets("/home/zqz/Work/chase/data/ml/w/labeled_data_all.csv",
#                       "/home/zqz/Work/chase/data/ml/ws-exp/labeled_data_all.csv",
#                       "/home/zqz/Work/chase/data/ml/w+ws/labeled_data_all.csv")

# in_folder="/home/zqz/Work/chase/data/annotation/unfiltered"
# out_file="/home/zqz/Work/chase/data/annotation/unfilered_merged.csv"
# in_folder="/home/zqz/Work/chase/data/annotation/keyword_filtered"
# out_file="/home/zqz/Work/chase/data/annotation/keywordfilered_merged.csv"
# print(os.getcwd())
# in_folder="../../../data/annotation/tag_filtered"
# out_file="../../../data/annotation/tagfilered_merged.csv"
# merge_annotations(in_folder,out_file)
