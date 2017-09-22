import csv
import pandas as pd


# racism=0, sexism=1,neither=2,both=3

def create_expert_corpus(out_file, in_file):
    with open(out_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["ds", "", "count", "hate_speech", "offensive_language,", "neither", "class", "tweet"])
        data = pd.read_csv(in_file, sep=',', encoding="utf-8")

        index = 0
        for row in data.itertuples():
            if index < 1:
                index += 1
                continue

            label = row[9]
            if label == 'racism':
                label = '0'
            elif label == 'sexism':
                label = '1'
            elif label == "both":
                label = '3'
            else:
                label = '2'

            writer.writerow([row[1],row[2],row[3],row[4],row[5],row[6],label, row[8]])
            index += 1
            if index % 100 == 0:
                print(index)


def create_amateur_corpus(out_file, in_file):
    with open(out_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["ds", "", "count", "hate_speech", "offensive_language,", "neither", "class", "tweet"])
        data = pd.read_csv(in_file, sep=',', encoding="utf-8")

        index = 0
        ties=0
        for row in data.itertuples():
            if index < 1:
                index += 1
                continue
            votes={}

            for i in range(10,13):
                annotation=row[i]
                if annotation in votes.keys():
                    count=votes[annotation]
                else:
                    count=0
                count+=1
                votes[annotation]=count

            highest = find_highest(votes)

            if len(highest)>1:
                label='2'
                ties+=1
            elif highest[0] == 'racism':
                label = '0'
            elif highest[0] == 'sexism':
                label = '1'
            elif highest[0] == "both":
                label = '3'
            else:
                label = '2'

            writer.writerow([row[1],row[2],row[3],row[4],row[5],row[6],label, row[8]])
            index += 1
            if index % 100 == 0:
                print(index)
        print("ties="+str(ties))


def find_highest(votes:dict):
    max=0
    for k in votes.keys():
        v = votes[k]
        if v>max:
            max=v

    res=[]
    for k in votes.keys():
        v = votes[k]
        if v==max:
            res.append(k)

    return res



def create_weighted_vote_corpus(out_file, in_file):
     with open(out_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["ds", "", "count", "hate_speech", "offensive_language,", "neither", "class", "tweet"])
        data = pd.read_csv(in_file, sep=',', encoding="utf-8")

        index = 0
        ties=0
        for row in data.itertuples():
            if index < 1:
                index += 1
                continue
            votes={}
            exp_vote=row[9]

            for i in range(9,13):
                annotation=row[i]
                if annotation in votes.keys():
                    count=votes[annotation]
                else:
                    count=0

                if i==9:
                    count+=2
                else:
                    count+=1
                votes[annotation]=count

            highest = find_highest(votes)

            if len(highest)>1:
                highest=[exp_vote]
                ties+=1

            if highest[0] == 'racism':
                label = '0'
            elif highest[0] == 'sexism':
                label = '1'
            elif highest[0] == "both":
                label = '3'
            else:
                label = '2'

            writer.writerow([row[1],row[2],row[3],row[4],row[5],row[6],label, row[8]])
            index += 1
            if index % 100 == 0:
                print(index)
        print("ties="+str(ties))


#####################
# create_expert_corpus("/home/zqz/Work/chase/data/ml/ws-exp/labeled_data_all.csv",
#                      "/home/zqz/GDrive/papers/chase/dataset/waseem2016/NLP+CSS_2016_tweets.csv")

# create_amateur_corpus("/home/zqz/Work/chase/data/ml/ws-amt/labeled_data_all.csv",
#                      "/home/zqz/GDrive/papers/chase/dataset/waseem2016/NLP+CSS_2016_tweets.csv")

create_weighted_vote_corpus("/home/zqz/Work/chase/data/ml/ws-merge/labeled_data_all.csv",
                     "/home/zqz/GDrive/papers/chase/dataset/waseem2016/NLP+CSS_2016_tweets.csv")
