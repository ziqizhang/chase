import random

import pandas as pd
import re
from nltk import PorterStemmer
from ml import text_preprocess as tp


def index_data(file_input, tweet_col, label_col):
    stemmer = PorterStemmer()
    raw_data = pd.read_csv(file_input, sep=',', encoding="utf-8")
    label_instances = {}
    stem_to_token = {}
    token_to_tweet = {}
    stem_dist = {}

    index = -1
    for row in raw_data.iterrows():
        if index%100==0: print(index)
        index += 1
        tweet = row[1][tweet_col]
        label = row[1][label_col]
        if label in label_instances.keys():
            label_instances[label] += 1
        else:
            label_instances[label] = 1

        tokens = tweet.split()

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

    return stem_dist, label_instances, stem_to_token, token_to_tweet, raw_data


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
        for v in stats.values():
            sum+=v

        #calculate token's score for each label (its frequency found as label 1, and 2, divide by...)
        label_one_count = 0
        if label_pair[0] in stats.keys():
            label_one_count=stats[label_pair[0]]
        token_scores_label_one[tok] = label_one_count / sum / label_percent[label_pair[0]]
        label_two_count = 0
        if label_pair[1] in stats.keys():
            label_two_count=stats[label_pair[1]]
        token_scores_label_two[tok] = label_two_count / sum / label_percent[label_pair[1]]

    return token_scores_label_one, token_scores_label_two


def replace_and_create(stems_to_replace: list, label_to_replace,
                       stem_to_token: dict, token_to_tweets: dict,
                       replacing_stems: list, replacing_label,
                       raw_data: pd.DataFrame):
    # go through eah stem
    generated_tweets = []
    for i in range(0, len(stems_to_replace)):
        s = stems_to_replace[i]
        # tokens that has the form of this stem
        if not s[0] in stem_to_token.keys():
            continue
        tokens = stem_to_token[s[0]]

        # generate new tweets by replacing
        for t in tokens:
            if t in token_to_tweets.keys():
                tweets_indices_contain_token = token_to_tweets[t]
                tweets_indices_matching_label = []
                for rt in tweets_indices_contain_token:
                    t_row = raw_data.ix[rt]
                    if t_row['class'] == label_to_replace:
                        tweets_indices_matching_label.append(rt)
                generated_tweets.extend(
                    replace_token(i, t, tweets_indices_matching_label,
                                  replacing_stems, replacing_label,
                                  stem_to_token, token_to_tweets, raw_data))


def replace_token(rank_in_list, token_to_replace,
                  tweets_indices_to_replace,
                  replacing_stems,
                  replacing_label, stem_to_token: dict,
                  token_to_tweets,
                  df: pd.DataFrame):
    # replace using the same-ranked stem and tokens
    replacing_stem = replacing_stems[rank_in_list]
    if not replacing_stem[0] in stem_to_token.keys():
        return []
    replacing_tokens = list(stem_to_token[replacing_stem[0]])
    selected_replacing_token = random.choice(replacing_tokens)

    for tok in selected_replacing_token:
        if not tok in token_to_tweets.keys():
            continue
        tweets = tweets_indices_to_replace[tok]
        for twt_id in tweets:
            twt_row = df[twt_id]
            new_row = list(twt_row)
            new_row['class'] = replacing_label
            new_row[0] = "mix"
            # replace tok in twt with token_to_replace
            replaced = re.sub('[' + token_to_replace + ']', tok, twt_row['tweet'])

            # todo:add new row

    generated_data = []


def rank_and_select_top(dict_with_scores: dict, topn):
    sorted_list = sorted(dict_with_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_list[0:topn]


if __name__ == "__main__":
    input_data = "/home/zz/Work/chase/data/ml/ml/w/labeled_data_all.csv"
    output_data = "/home/zz/Work/chase/data/ml/ml/w/labeled_data_all_mixed.csv"
    data_col = 7

    # read data
    stem_dist, label_instances, stem_to_token, token_to_tweet, raw_data \
        = index_data(input_data, 7, 6)

    token_scores_label_one, token_scores_label_two\
        =score_tokens(stem_dist, label_instances, [0, 1])

    # find the two classes to be mixed
    ranked_label1_stems=rank_and_select_top(token_scores_label_one, 100)
    ranked_label2_stems=rank_and_select_top(token_scores_label_two, 100)

    #replace tweets of label 1 by tokens from label 2
    replace_and_create(ranked_label1_stems, 0, stem_to_token,
                       token_to_tweet,
                       ranked_label2_stems, 1, raw_data)

    #replace tweets of label 2 by tokens from label 1

    # for each class, analyse each tweet
    # build look up table
    # build word distribution

    # rank words

    # swap data

    print("end")
