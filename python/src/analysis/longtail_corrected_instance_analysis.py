import csv

import os
import pandas as pd
from ml import classifier_dnn as cd


# for each feature belonging to each class, calculate its distribution score, which is:
# freq(f1, c1)/#c1 / freq(f1, non-c1)/#non-c1
def calc_instance_unique_feature_percent(input_data_file, sys_out,
                                         word_norm_option, label_col):
    raw_data = pd.read_csv(input_data_file, sep=',', encoding="utf-8")
    # call method to get vocabulary. need to change this method for n-grams
    M = cd.get_word_vocab(raw_data.tweet, sys_out, word_norm_option)
    # M=self.feature_scale(M)

    class_unique_features = {}  # key - class labe; value-set of features for that class
    feature_dist_ovr_class = {}  # key-feature; value-dict containinng the feature's frequency found in each class
    # stats={}
    class_instance_count = {}
    class_row_index_map = {}  # key-label; value-set of indexes in the data frame pointing to the tweets belonging to that class
    M0 = M[0]
    inverted_dict = dict([(v, k) for k, v in M[1].items()])
    for index, row in raw_data.iterrows():
        # print(index)
        vocab = M0[index]  # get the vocab for that tweet, only indices are returned
        label = row[label_col]  # get the label for that tweet

        if label in class_instance_count.keys():
            class_instance_count[label] += 1
        else:
            class_instance_count[label] = 1

        if label in class_row_index_map.keys():
            class_row_index_map[label].add(index)
        else:
            indices = set()
            indices.add(index)
            class_row_index_map[label] = indices

        for v in vocab:
            string = inverted_dict[v]
            # update frequency of this feature found in this class label
            if string in feature_dist_ovr_class.keys():
                dist = feature_dist_ovr_class[string]
            else:
                dist = {}
            if label in dist.keys():
                dist[label] += 1
            else:
                dist[label] = 1
                feature_dist_ovr_class[string] = dist

    # work out unique features per class
    for ft, dist in feature_dist_ovr_class.items():
        if len(dist) == 1:
            label = list(dist.keys())[0]
            if label in class_unique_features.keys():
                class_fs = class_unique_features[label]
            else:
                class_fs = set()
            class_fs.add(ft)
            class_unique_features[label] = class_fs

    # map between index of a tweet in the dataset and the segment of long tail it resides
    tweet_longtail_segment = {}
    for index, row in raw_data.iterrows():
        # print(index)
        vocab = M0[index]  # get the vocab for that tweet, only indices are returned
        label = row[label_col]

        if len(vocab) == 0:
            continue

        score_uniqueness = 0
        if label in class_unique_features.keys():
            class_unique_fs = class_unique_features[label]
            all_fs = 0
            unique_fs = 0
            for v in vocab:
                string = inverted_dict[v]
                if string in class_unique_fs:
                    unique_fs += 1
                all_fs += 1

            score_uniqueness = unique_fs / all_fs

        if score_uniqueness == 0:
            tweet_longtail_segment[index] = "0"
        elif score_uniqueness > 0 and score_uniqueness < 0.1:
            tweet_longtail_segment[index] = "(0-0.1)"
        elif score_uniqueness >= 0.1 and score_uniqueness < 0.2:
            tweet_longtail_segment[index] = "[0.1-0.2)"
        elif score_uniqueness >= 0.2 and score_uniqueness < 0.3:
            tweet_longtail_segment[index] = "[0.2-0.3)"
        elif score_uniqueness >= 0.3 and score_uniqueness < 0.4:
            tweet_longtail_segment[index] = "[0.3-0.4)"
        elif score_uniqueness >= 0.4 and score_uniqueness < 0.5:
            tweet_longtail_segment[index] = "[0.4-0.5)"
        elif score_uniqueness >= 0.5 and score_uniqueness < 0.6:
            tweet_longtail_segment[index] = "[0.5-0.6)"
        elif score_uniqueness >= 0.6 and score_uniqueness < 0.7:
            tweet_longtail_segment[index] = "[0.6-0.7)"
        elif score_uniqueness >= 0.7 and score_uniqueness < 0.8:
            tweet_longtail_segment[index] = "[0.7-0.8)"
        elif score_uniqueness >= 0.8 and score_uniqueness < 0.9:
            tweet_longtail_segment[index] = "[0.8-0.9)"
        elif score_uniqueness >= 0.9:
            tweet_longtail_segment[index] = "[0.9-1.0]"

    return tweet_longtail_segment


def find_corrected_tweets(ref_annotation_output, impr_annotation_output,
                          tweet_longtail_segment,
                          raw_input_data_file, label_col):
    raw_data = pd.read_csv(raw_input_data_file, sep=',', encoding="utf-8")
    ref_annotation = pd.read_csv(ref_annotation_output, sep=',', encoding="utf-8")
    impr_annotation = pd.read_csv(impr_annotation_output, sep=',', encoding="utf-8")

    correction_stats = {}
    correction_stats["0"]=0
    correction_stats["(0-0.1)"] = 0
    correction_stats["[0.1-0.2)"] = 0
    correction_stats["[0.2-0.3)"] = 0
    correction_stats["[0.3-0.4)"] = 0
    correction_stats["[0.4-0.5)"] = 0
    correction_stats["[0.5-0.6)"] = 0
    correction_stats["[0.6-0.7)"] = 0
    correction_stats["[0.7-0.8)"] = 0
    correction_stats["[0.8-0.9)"] = 0
    correction_stats["[0.9-1.0]"] = 0


    for index in range(0, len(ref_annotation)):
        if index not in tweet_longtail_segment.keys():
            continue
        # print(index)
        #label = row[label_col]

        # get ref annotation
        ref_ann = ref_annotation.loc[index]
        impr_ann = impr_annotation.loc[index]

        if not ref_ann[0]==impr_ann[0]:
            print("index on the same row of two data files are not the same...")

        if impr_ann[2] == "ok" and ref_ann[2] == "wrong":
            print(ref_ann[0])
            # this is a corrected tweet by impr'ed model
            longtail_segment = tweet_longtail_segment[index]
            if longtail_segment in correction_stats.keys():
                correction_stats[longtail_segment] += 1
            else:
                correction_stats[longtail_segment] = 1

    return correction_stats


def generate_stats(input_data_file, sys_out,
                   output_data_file, word_norm_option, label_col,
                   ref_annotation_output,
                   impr_annotation_output,):
    tweet_longtail_segment=\
        calc_instance_unique_feature_percent(input_data_file,sys_out,
                                             word_norm_option, label_col)
    correction_stats=find_corrected_tweets(ref_annotation_output, impr_annotation_output,
                          tweet_longtail_segment, input_data_file,
                          label_col)

    keylist = correction_stats.keys()
    with open(output_data_file, 'w', newline='\n') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                               quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for key in sorted(keylist):
            csvwriter.writerow([key, correction_stats[key]])


if __name__ == "__main__":
    #ref_annotation = pd.read_csv("/home/zz/Work/chase/output/errors/errors.csv", sep=',', encoding="utf-8")
    #ref_annotation.sort_values(ref_annotation.columns[0])

    input_data = "/home/zz/Work/chase/data/ml/ml/w+ws/labeled_data_all.csv"
    ref_ann_folder="/home/zz/Work/chase/output/errors/entire_dataset/w+ws/base"
    impr_skip_ann_folder="/home/zz/Work/chase/output/errors/entire_dataset/w+ws/base_scnn"
    impr_gru_ann_folder="/home/zz/Work/chase/output/errors/entire_dataset/w+ws/base_gru"
    sys_out = "/home/zz/Work/chase/output"
    word_norm_option = 0
    label_col = 6

    ref_ann_files=sorted(os.listdir(ref_ann_folder))
    impr_skip_ann_files = sorted(os.listdir(impr_skip_ann_folder))
    impr_gru_ann_files = sorted(os.listdir(impr_gru_ann_folder))

    for rf, skipf, gruf in zip(ref_ann_files, impr_skip_ann_files, impr_gru_ann_files):
        print("rf={}\nskipf={}".format(rf,skipf))
        generate_stats(input_data, sys_out,
                       sys_out+"/skip_vs_base-{}".format(skipf), word_norm_option, label_col,
                       ref_ann_folder+"/"+rf,
                       impr_skip_ann_folder+"/"+skipf)
        print("rf={}\ngruf={}".format(rf, gruf))
        generate_stats(input_data, sys_out,
                       sys_out + "/gru_vs_base-{}".format(gruf), word_norm_option, label_col,
                       ref_ann_folder + "/" + rf,
                       impr_gru_ann_folder + "/" + gruf)