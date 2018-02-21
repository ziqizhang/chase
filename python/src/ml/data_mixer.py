import csv
import random

import numpy
import pandas as pd
import re
from nltk import PorterStemmer
import nltk
from ml import text_preprocess as tp


def index_data(file_input, tweet_col, label_col):
    stemmer = PorterStemmer()
    raw_data = pd.read_csv(file_input, sep=',', encoding="utf-8")
    label_instances = {}
    stem_to_token = {}
    token_to_tweet = {}
    stem_dist = {}
    label_to_tweet_stems={}

    index = -1
    for row in raw_data.iterrows():
        if index % 100 == 0: print(index)
        index += 1
        tweet = row[1][tweet_col]
        label = row[1][label_col]
        if label in label_instances.keys():
            label_instances[label] += 1
        else:
            label_instances[label] = 1

        tokens = tweet.split()

        stems=set()
        for ot in tokens:
            if ot in token_to_tweet.keys():
                token_to_tweet[ot].add(index)
            else:
                tweets = set()
                tweets.add(index)
                token_to_tweet[ot] = tweets

            t = tp.strip_hashtags(ot)
            t = " ".join(re.split("[^a-zA-Z]*", t.lower())).strip()
            t = stemmer.stem(t)
            if len(t) > 0:
                stems.add(t)
                if t in stem_to_token.keys():
                    stem_to_token[t].add(ot)
                else:
                    tokens = set()
                    tokens.add(ot)
                    stem_to_token[t] = tokens

                # update stats about the distribution of the stem
                if t in stem_dist.keys():
                    dist_stats = stem_dist[t]
                else:
                    dist_stats = {}
                if label in dist_stats.keys():
                    dist_stats[label] += 1
                else:
                    dist_stats[label] = 1
                stem_dist[t] = dist_stats

        if label in label_to_tweet_stems.keys():
            tweet_stems=label_to_tweet_stems[label]
        else:
            tweet_stems=dict()
        tweet_stems[index]=stems
        label_to_tweet_stems[label]=tweet_stems

    return stem_dist, label_instances, stem_to_token, token_to_tweet, raw_data, label_to_tweet_stems


def score_tokens(token_dist: dict, label_instances: dict, label_pair: list):
    token_scores_label_one = {}
    token_scores_label_two = {}

    label_percent = {}
    sum = 0
    for label, instances in label_instances.items():
        sum += instances
    for label, instances in label_instances.items():
        label_percent[label] = instances / sum

    for tok, stats in token_dist.items():
        sum = 0
        # for l in label_pair:
        #     if l in stats.keys():
        #         sum += stats[l]
        # if sum==0:
        #     continue
        for v in stats.values(): #stats contains frequency of this tok found in all labels
            sum += v

        # calculate token's score for each label (its frequency found as label 1, and 2, divide by...)
        label_one_count = 0
        if label_pair[0] in stats.keys():
            label_one_count = stats[label_pair[0]] #label one count is freq of this tok found as this label one
        token_scores_label_one[tok] = label_one_count / sum # / label_percent[label_pair[0]]
        label_two_count = 0
        if label_pair[1] in stats.keys():
            label_two_count = stats[label_pair[1]]
        token_scores_label_two[tok] = label_two_count / sum #/ label_percent[label_pair[1]]

    return token_scores_label_one, token_scores_label_two

#given tweets of two classes, replace words of tweets from one class with words from another class
def replace_and_create_acrossclass(tweet_to_replace_and_stem,
                                   nouns_to_replace,
                                   nouns_replace_by,
                                   adjs_to_replace,
                                   adjs_replace_by,
                                   stem_to_token,
                                   new_label,
                                   raw_data):
    # go through eah stem
    generated_tweets = []
    for tw, stems in tweet_to_replace_and_stem.items():
        new_row = pd.Series.copy(raw_data.ix[tw])
        tweet_id = new_row[1]
        tweet = new_row["tweet"]
        replaced = False
        for st in stems:
            if st in nouns_to_replace:
                replaced, tweet = regex_replace(st,
                                                nouns_replace_by, stem_to_token,
                                                tweet)
            elif st in adjs_to_replace:
                replaced, tweet = regex_replace(st,
                                                adjs_replace_by, stem_to_token,
                                                tweet)

        if replaced:
            new_row[0] = "mix"
            new_row["class"] = new_label
            new_row["tweet"] = tweet

        generated_tweets.append(new_row)

    return generated_tweets

#for a given class, take a tweet, replace randomly some words with other words found in this class of tweets
def replace_and_create_singleclass(tweet_to_replace_and_stem,
                                   nouns_replace_by,
                                   adjs_replace_by,
                                   stem_to_token,
                                   raw_data):
    # go through eah stem
    generated_tweets = []
    for tw, stems in tweet_to_replace_and_stem.items():
        new_row = pd.Series.copy(raw_data.ix[tw])
        tweet_id=new_row[1]

        tweet = new_row["tweet"]
        tokens = tweet.split()
        #try:
        tags = nltk.pos_tag(tokens)
        # except IndexError:
        #     print("")

        noun_indices = []
        adj_indices = []
        for i in range(0, len(tags)):
            if "NN" in tags[i]:
                noun_indices.append(i)
            if "J" in tags[i]:
                adj_indices.append(i)
        num_n = int(len(noun_indices) / 2)
        num_a = int(len(adj_indices) / 2)

        # replace nouns
        for j in range(0, num_n):
            tok_to_replace = tokens[random.choice(noun_indices)]
            tweet = regex_replace_single(tok_to_replace,
                                            nouns_replace_by, stem_to_token,
                                            tweet)

        # replace adj
        for j in range(0, num_a):
            tok_to_replace = tokens[random.choice(adj_indices)]
            tweet = regex_replace_single(tok_to_replace,
                                            adjs_replace_by, stem_to_token,
                                            tweet)

        new_row[0] = "mix"
        new_row["tweet"] = tweet

        generated_tweets.append(new_row)

    return generated_tweets


def regex_replace(stem, stems_replace_by,
                  stem_to_token: dict, tweet):
    tokens = stem_to_token[stem]
    stem_replace_by = random.choice(stems_replace_by)
    replaced = False
    for tok in tokens:
        tok_replace_by = list(stem_to_token[stem_replace_by])[0]
        # replace tok by
        insensitive_regex = re.compile(re.escape(tok), re.IGNORECASE)
        tweet = insensitive_regex.sub(tok_replace_by, tweet)
        replaced = True
    return replaced, tweet


def regex_replace_single(tok, stems_replace_by,
                  stem_to_token: dict, tweet):
    stem_replace_by = random.choice(stems_replace_by)
    tok_replace_by = list(stem_to_token[stem_replace_by])[0]
        # replace tok by
    insensitive_regex = re.compile(re.escape(tok), re.IGNORECASE)
    tweet = insensitive_regex.sub(tok_replace_by, tweet)
    return tweet

def rank_and_select_top(dict_with_scores: dict, topn):
    sorted_list = sorted(dict_with_scores.items(), key=lambda x: x[1], reverse=True)
    if topn>0:
        return sorted_list[0:topn]
    else:
        sublist=[]
        for e in sorted_list:
            if e[1]==1.0:
                sublist.append(e)
        return sublist


def postag_stems(ranked_stems, stem_to_token):
    verbs = []
    nouns = []
    adjs = []
    for s in ranked_stems:
        tokens = list(stem_to_token[s[0]])
        tag = nltk.pos_tag([tokens[0]])[0]
        if "NN" in tag[1]:
            nouns.append(s[0])
        elif "V" in tag[1]:
            verbs.append(s[0])
        elif "J" in tag[1]:
            adjs.append(s[0])
    return verbs, nouns, adjs


def map_tweet_to_stem(ranked_label1_stems: list,
                      stem_to_token: dict,
                      token_to_tweet: dict):
    map = {}
    for s in ranked_label1_stems:
        tokens = stem_to_token[s[0]]
        for t in tokens:
            tweets = token_to_tweet[t]
            for tw in tweets:
                if tw in map.keys():
                    map[tw].add(s[0])
                else:
                    stems = set()
                    stems.add(s[0])
                    map[tw] = stems
    return map


def write_to_file(generated_tweets, raw_data, out_file):
    merged = raw_data.as_matrix()
    merged = numpy.concatenate((merged, generated_tweets), axis=0)

    header = [list(raw_data.columns.values)]
    generated_tweets = numpy.concatenate((header, generated_tweets), axis=0)
    with open(out_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                               quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in generated_tweets:
            csvwriter.writerow(list(row))


if __name__ == "__main__":
    input_data = "/home/zz/Work/chase/data/ml/ml/w/labeled_data_all.csv"
    output_data = "/home/zz/Work/chase/data/ml/ml/w/labeled_data_all_mixed_debug.csv"
    #input_data = "/home/zz/Work/chase/data/ml/ml/ws-amt/labeled_data_all.csv"
    #output_data = "/home/zz/Work/chase/data/ml/ml/ws-amt/labeled_data_all_mixed.csv"
    #input_data = "/home/zz/Work/chase/data/ml/ml/ws-exp/labeled_data_all.csv"
    #output_data = "/home/zz/Work/chase/data/ml/ml/ws-exp/labeled_data_all_mixed.csv"
    #input_data = "/home/zz/Work/chase/data/ml/ml/ws-gb/labeled_data_all.csv"
    #output_data = "/home/zz/Work/chase/data/ml/ml/ws-gb/labeled_data_all_mixed.csv"
    #input_data = "/home/zz/Work/chase/data/ml/ml/w+ws/labeled_data_all.csv"
    #output_data = "/home/zz/Work/chase/data/ml/ml/w+ws/labeled_data_all_mixed.csv"

    #input_data = "/home/zz/Work/chase/data/ml/ml/w+ws/labeled_data_all.csv"
    #output_data = "/home/zz/Work/chase/data/ml/ml/w+ws/labeled_data_all_mixed_single.csv"
    #input_data = "/home/zz/Work/chase/data/ml/ml/rm/labeled_data_all.csv"
    #output_data = "/home/zz/Work/chase/data/ml/ml/rm/labeled_data_all_mixed_single.csv"
    # input_data = "/home/zz/Work/chase/data/ml/ml/dt/labeled_data_all_2.csv"
    # output_data = "/home/zz/Work/chase/data/ml/ml/dt/labeled_data_all_2_mixed_single.csv"

    #
    tweet_col = 7
    label_col=6
    label_pair=[0,1]

    # read data
    stem_dist, label_instances, stem_to_token, token_to_tweet, raw_data, label_to_tweetstems \
        = index_data(input_data, tweet_col, label_col)

    token_scores_label_one, token_scores_label_two \
        = score_tokens(stem_dist, label_instances, label_pair)

    # find the two classes to be mixed
    ranked_label1_stems = rank_and_select_top(token_scores_label_one, 0)
    # label 1 stem verbs, nouns
    selected_label1_verbs, selected_label1_nouns, selected_label1_adjs = \
        postag_stems(ranked_label1_stems, stem_to_token)

    ranked_label2_stems = rank_and_select_top(token_scores_label_two, 0)
    # label 2 stem verbs, nouns
    selected_label2_verbs, selected_label2_nouns, selected_label2_adjs = \
        postag_stems(ranked_label2_stems, stem_to_token)

    # map tweet to list of stems
    # tweet_to_stem_label1 = \
    #     map_tweet_to_stem(ranked_label1_stems, stem_to_token, token_to_tweet)
    # tweet_to_stem_label2 = \
    #     map_tweet_to_stem(ranked_label2_stems, stem_to_token, token_to_tweet)
    tweet_to_stem_label1 = label_to_tweetstems[0]
    tweet_to_stem_label2 = label_to_tweetstems[1]

    # replace tweets of label 1 by tokens from label 2
    new_data_label1 = replace_and_create_acrossclass(tweet_to_stem_label1,
                                          selected_label1_nouns,
                                          selected_label2_nouns,
                                          selected_label1_adjs,
                                          selected_label2_adjs,
                                          stem_to_token, 1, raw_data)
    print("label {} original data={}, newly generated={}".format(1, len(tweet_to_stem_label1),
                                                                  len(new_data_label1)))
    #
    new_data_label2 = replace_and_create_acrossclass(tweet_to_stem_label2,
                                          selected_label2_nouns,
                                          selected_label1_nouns,
                                          selected_label2_adjs,
                                          selected_label1_adjs,
                                          stem_to_token, 0, raw_data)

    # new_data_label1 = replace_and_create_singleclass(tweet_to_stem_label1,
    #                                                  selected_label1_nouns,
    #                                                  selected_label1_adjs,
    #                                                  stem_to_token, raw_data)
    # print("label {} original data={}, newly generated={}".format(1, len(tweet_to_stem_label1),
    #                                                              len(new_data_label1)))
    #
    # new_data_label2 = replace_and_create_singleclass(tweet_to_stem_label2,
    #                                                  selected_label2_nouns,
    #                                                  selected_label2_adjs,
    #                                                  stem_to_token, raw_data)
    # print("label {} original data={}, newly generated={}".format(0, len(tweet_to_stem_label2),
    #                                                              len(new_data_label2)))
    new_data_label1.extend(new_data_label2)
    write_to_file(new_data_label1, raw_data, output_data)

    print("end")
