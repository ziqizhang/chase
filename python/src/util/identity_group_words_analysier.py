'''
This file is created to analyse the correlation between
- presence of identity group words (see https://www.aclweb.org/anthology/2020.acl-main.483.pdf)
this list of 25 words are here: /home/zz/Work/data/identity_group_words.txt
- sentiment of the text containing that igw
- whether it is hate or not
'''
import pandas as pd
import re, csv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sa = SentimentIntensityAnalyzer()



#the following are a list of functions for reading different input data
def read_AG10(in_file, header:int, delimiter:str, text_col:int, label_col:int):
    df = pd.read_csv(in_file, header=header, delimiter=delimiter, quotechar='"', encoding="utf-8")
    df = df.fillna('')
    df = df.as_matrix()

    data=[]
    for row in df:
        if row[label_col].lower() == "nag":
            data.append([row[text_col], "no"])
        else:
            data.append([row[text_col], "yes"])

    return data

def read_multilabel(in_file, header:int, delimiter:str, text_col:int, label_col_start:int):
    df = pd.read_csv(in_file, header=header, delimiter=delimiter, quotechar='"', encoding="utf-8")
    df = df.fillna('')
    df = df.as_matrix()

    data = []
    for row in df:
        labels = row[label_col_start:]
        if 1 in labels:
            data.append([row[text_col], "yes"])
        else:
            data.append([row[text_col], "no"])

    return data

def read_tweet50k(in_file, header:int, delimiter:str, text_col:int, label_col:int):
    df = pd.read_csv(in_file, header=header, delimiter=delimiter, quotechar='"', encoding="utf-8")
    df = df.fillna('')
    df = df.as_matrix()

    data = []
    for row in df:
        if row[label_col].lower() == "normal":
            data.append([row[text_col], "no"])
        else:
            data.append([row[text_col], "yes"])

    return data

def read_waseem(in_file, header:int, delimiter:str, text_col:int, label_col:int):
    df = pd.read_csv(in_file, header=header, delimiter=delimiter, quotechar='"', encoding="utf-8")
    df = df.fillna('')
    df = df.as_matrix()

    data = []
    for row in df:
        if row[label_col] == 2:
            data.append([row[text_col], "no"])
        else:
            data.append([row[text_col], "yes"])

    return data

#read the list of identity group words
def read_igw(in_file):
    f = open(in_file, 'r')
    line = f.readline()
    res=set()
    while line:
        line = f.readline().strip()
        if line.startswith("#") or len(line)==0:
            continue
        res.add(line)
    return res

#filter data based on the identity group words
def filter_by_igw(igw:set, data:list):
    true_pos_with_igw=[]
    true_neg_with_igw=[]
    total=0
    totalPos=0
    totalNeg=0

    pos_with_igw=0
    neg_with_igw=0

    for r in data:
        total+=1
        if r[1]=="yes":
            totalPos+=1
        else:
            totalNeg+=1

        words = set(re.split("[^a-zA-Z]*", r[0].lower()))
        inter = words.intersection(igw)

        if len(inter)>0 and r[1]=="yes":
            pos_with_igw+=1
            true_pos_with_igw.append(r[0])

        elif len(inter) > 0 and r[1] == "no":
            neg_with_igw += 1
            true_neg_with_igw.append(r[0])

    print('Total instances, {}\nTotal pos, {} \nTotal neg, {} \n'
          'Total pos with IDW, {} \nTotal neg with IDW, {}'.format(total, totalPos, totalNeg,
                                                                    pos_with_igw, neg_with_igw))

    return true_pos_with_igw, true_neg_with_igw


#create sentiment distribution data based on the filtered data by the function above
#the data will be formatted for boxplot analysis
def calc_sentiment(text:list, out_file):
    with open(out_file, 'w', newline='\n') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(["text","polarity"])
        count=0
        for t in text:
            count+=1
            #print(count)
            polar=sa.polarity_scores(t)
            t = " ".join(re.split("[^a-zA-Z]*", t.lower())).strip()
            csvwriter.writerow([t, polar["compound"]])

if __name__ == '__main__':
    idgw_file="/home/zz/Work/data/toxic_content/identity_group_words.txt"
    igw=read_igw(idgw_file)

    #AG10
    print("Dataset = AG10")
    in_file="/home/zz/Work/data/toxic_content/AG10K_train.csv"
    data=read_AG10(in_file,0,",",1,2)
    pos_with_igw, neg_with_igw = filter_by_igw(igw, data)
    out_file = "/home/zz/Work/data/toxic_content/stats/AG10K_train_stats_pos.csv"
    calc_sentiment(pos_with_igw, out_file)
    out_file = "/home/zz/Work/data/toxic_content/stats/AG10K_train_stats_neg.csv"
    calc_sentiment(neg_with_igw, out_file)

    #Multilabel
    print("Dataset = multilabel")
    in_file = "/home/zz/Work/data/toxic_content/multi-label_train.csv"
    data = read_multilabel(in_file, 0, ",", 2, 3)
    pos_with_igw, neg_with_igw = filter_by_igw(igw, data)
    out_file = "/home/zz/Work/data/toxic_content/stats/multilabel_train_stats_pos.csv"
    calc_sentiment(pos_with_igw, out_file)
    out_file = "/home/zz/Work/data/toxic_content/stats/multilabel_train_stats_neg.csv"
    calc_sentiment(neg_with_igw, out_file)

    #tweet50k
    # print("Dataset = tweet50k")
    # in_file = "/home/zz/Work/data/toxic_content/tweet50k_train.csv"
    # data = read_tweet50k(in_file, 0, ",", 3, 2)
    # pos_with_igw, neg_with_igw = filter_by_igw(igw, data)
    # out_file = "/home/zz/Work/data/toxic_content/stats/tweet50k_train_stats_pos.csv"
    # calc_sentiment(pos_with_igw, out_file)
    # out_file = "/home/zz/Work/data/toxic_content/stats/tweet50k_train_stats_neg.csv"
    # calc_sentiment(neg_with_igw, out_file)

    # waseem
    print("Dataset = waseem")
    in_file = "/home/zz/Work/data/toxic_content/wassem_train.csv"
    data = read_waseem(in_file, 0, ",", 2, 1)
    pos_with_igw, neg_with_igw = filter_by_igw(igw, data)
    out_file = "/home/zz/Work/data/toxic_content/stats/wassem_train_stats_pos.csv"
    calc_sentiment(pos_with_igw, out_file)
    out_file = "/home/zz/Work/data/toxic_content/stats/wassem_train_stats_neg.csv"
    calc_sentiment(neg_with_igw, out_file)