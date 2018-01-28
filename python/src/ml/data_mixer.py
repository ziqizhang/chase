import pandas as pd
import re
from nltk import PorterStemmer
from ml import text_preprocess as tp

def index_data(file_input, data_col, label_col):
    stemmer=PorterStemmer()
    raw_data = pd.read_csv(file_input, sep=',', encoding="utf-8")
    label_instances={}
    stem_to_token={}
    token_to_tweet={}
    stem_dist={}

    index=-1
    for row in raw_data:
        index+=1
        tweet=row[data_col]
        label=row[label_col]
        if label in label_instances.keys():
            label_instances[label]+=1
        else:
            label_instances[label]=1

        tokens = tweet.split()

        for ot in tokens:
            if ot in token_to_tweet.keys():
                token_to_tweet[ot].add(index)
            else:
                tweets=set()
                tweets.add(index)
                token_to_tweet[ot]=tweets

            t=tp.strip_hashtags(ot)
            t=" ".join(re.split("[^a-zA-Z]*", t.lower())).strip()
            t=stemmer.stem(t)
            if len(t)>0:
                if t in stem_to_token.keys():
                    stem_to_token[t].add(ot)
                else:
                    tokens=set()
                    tokens.add(ot)
                    stem_to_token[t]=tokens


                #update stats about the distribution of the stem
                if t in stem_dist.keys():
                    dist_stats=stem_dist[t]
                else:
                    dist_stats={}
                if label in dist_stats.keys():
                    dist_stats[label]+=1
                else:
                    dist_stats[label]=1
                stem_dist[t]=dist_stats

    return stem_dist, label_instances, stem_to_token, token_to_tweet

def rank_tokens(token_dist:dict, label_instances:dict, label_pair:list):
    token_scores_label_one={}
    token_scores_label_two={}

    label_percent={}
    sum=0
    for label, instances in label_instances.items():
        sum+=len(instances)
    for label, instances in label_instances.items():
        label_percent[label]=len(instances)/sum

    for tok, stats in token_dist.items():
        sum=0
        for l in label_pair:
            sum+=stats[l]

        label_one_count=stats[label_pair[0]]
        token_scores_label_one[tok]=label_one_count/sum/label_percent[label_pair[0]]
        label_two_count = stats[label_pair[1]]
        token_scores_label_two[tok] = label_two_count / sum / label_percent[label_pair[1]]




if __name__ == "__main__":
    input_data = "/home/zz/Work/chase/data/ml/ml/w/labeled_data_all.csv"
    output_data = "/home/zz/Work/chase/data/ml/ml/w/labeled_data_all_mixed.csv"
    data_col=7


    #read data

    #find the two classes to be mixed

    #for each class, analyse each tweet
        #build look up table
        #build word distribution


    #rank words

    #swap data


    print("end")