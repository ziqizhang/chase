import csv

from ml import classifier_dnn as cd
import pandas as pd


# for each feature belonging to each class, calculate its distribution score, which is:
# freq(f1, c1)/#c1 / freq(f1, non-c1)/#non-c1
def calc_feature_score_distribution(input_data_file, sys_out, output_data_folder, word_norm_option, label_col):
    raw_data = pd.read_csv(input_data_file, sep=',', encoding="utf-8")
    # call method to get vocabulary. need to change this method for n-grams
    M = cd.get_word_vocab(raw_data.tweet, sys_out, word_norm_option)
    # M=self.feature_scale(M)

    class_features = {}  # key - class labe; value-set of features for that class
    feature_dist_ovr_class = {}  # key-feature; value-dict containinng the feature's frequency found in each class
    # stats={}
    class_instance_count = {}
    M0 = M[0]
    inverted_dict = dict([(v, k) for k, v in M[1].items()])
    for index, row in raw_data.iterrows():
        # print(index)
        vocab = M0[index]  # get the vocab for that tweet, only indices are returned
        label = row[label_col]  # get the label for that tweet
        if label in class_features.keys():
            class_fs = class_features[label]
        else:
            class_fs = set()

        if label in class_instance_count.keys():
            class_instance_count[label] += 1
        else:
            class_instance_count[label] = 1

        for v in vocab:
            string = inverted_dict[v]
            class_fs.add(string)
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

        # update of features for this class complete, update dict
        class_features[label] = class_fs

    # saving to output files
    for k, v in class_features.items():
        label = k
        output_data_file = output_data_folder + "_" + str(label) + ".csv"
        with open(output_data_file, 'w', newline='\n') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',',
                                   quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(["label=" + str(label)])

            # go through features of this class, for each one, calculate a score
            # features_of_label = v
            for f in v:
                # calculate score of this feature for this label/class
                dist = feature_dist_ovr_class[f]
                if not label in dist.keys():
                    continue

                freq_in_c = dist[label]
                inst_in_c = class_instance_count[label]
                freq_in_nonc = 0
                for l, fr in dist.items():
                    if l == label:
                        continue
                    freq_in_nonc += fr
                inst_in_nonc = 0
                for l, insts in class_instance_count.items():
                    if l == label:
                        continue
                    inst_in_nonc += insts

                # score = freq_in_c/inst_in_c /((freq_in_nonc/inst_in_nonc) + 1)
                # score = freq_in_c / (freq_in_nonc + 1)
                score = freq_in_c / (freq_in_nonc + 1)
                if score >= 1:
                    score = score / inst_in_c
                    csvwriter.writerow([f, score])

    print("end")


# for each feature belonging to each class, calculate its distribution score, which is:
# freq(f1, c1)/#c1 / freq(f1, non-c1)/#non-c1
def calc_instance_unique_feature_percent(input_data_file, sys_out, output_data_file, word_norm_option, label_col):
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

    # next loop through each tweet, output its label feature uniqueness as |features(label)| / |features(*)
    # for cls, indices in class_row_index_map.items():
    #     output_data_file = output_data_folder + "_"+str(cls)+".csv"
    #     label=cls
    #     with open(output_data_file, 'w', newline='\n') as csvfile:
    #         csvwriter = csv.writer(csvfile, delimiter=',',
    #                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #         for index in indices:
    #             #print(index)
    #             vocab=M0[index] #get the vocab for that tweet, only indices are returned
    #             if len(vocab)==0:
    #                 continue
    #
    #             score_uniqueness=0
    #             if label in class_unique_features.keys():
    #                 class_unique_fs = class_unique_features[label]
    #
    #                 all_fs=0
    #                 unique_fs=0
    #                 for v in vocab:
    #                     string=inverted_dict[v]
    #                     if string in class_unique_fs:
    #                         unique_fs+=1
    #                     all_fs+=1
    #
    #                 score_uniqueness=unique_fs/all_fs
    #
    with open(output_data_file, 'w', newline='\n') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                                   quotechar='"', quoting=csv.QUOTE_MINIMAL)
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

            csvwriter.writerow([label, score_uniqueness])

    print("end")


# for each feature belonging to each class, calculate its distribution score, which is:
# freq(f1, c1)/#c1 / freq(f1, non-c1)/#non-c1
def calc_average_feature_uniqueness(input_data_file, sys_out, output_data_file, word_norm_option, label_col):
    raw_data = pd.read_csv(input_data_file, sep=',', encoding="utf-8")
    # call method to get vocabulary. need to change this method for n-grams
    M = cd.get_word_vocab(raw_data.tweet, sys_out, word_norm_option)
    # M=self.feature_scale(M)

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

    # next loop through each tweet, output its label average feature uniqueness
    # for cls, indices in class_row_index_map.items():
    #     output_data_file = output_data_folder + "_"+str(cls)+".csv"
    #     label=cls
    #     with open(output_data_file, 'w', newline='\n') as csvfile:
    #         csvwriter = csv.writer(csvfile, delimiter=',',
    #                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #         for index in indices:
    #             #print(index)
    #             vocab=M0[index] #get the vocab for that tweet, only indices are returned
    #             if len(vocab)==0:
    #                 continue
    #
    #             score_uniqueness=0
    #             if label in class_unique_features.keys():
    #                 class_unique_fs = class_unique_features[label]
    #
    #                 all_fs=0
    #                 unique_fs=0
    #                 for v in vocab:
    #                     string=inverted_dict[v]
    #                     if string in class_unique_fs:
    #                         unique_fs+=1
    #                     all_fs+=1
    #
    #                 score_uniqueness=unique_fs/all_fs
    #
    #             csvwriter.writerow([label,score_uniqueness])
    with open(output_data_file, 'w', newline='\n') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                               quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for index, row in raw_data.iterrows():
            # print(index)
            vocab = M0[index]  # get the vocab for that tweet, only indices are returned
            label = row[label_col]  # get the label for that tweet

            features = len(vocab)
            if features == 0:
                continue

            score_sum_uniqueness = 0
            for v in vocab:
                string = inverted_dict[v]
                fs_dist_ovr_cls = feature_dist_ovr_class[string]
                fs_for_label = fs_dist_ovr_cls[label]
                inst_for_label = class_instance_count[label]

                fs_for_other_label = 0
                inst_for_other_label = 0
                for l, d in fs_dist_ovr_cls.items():
                    if label == l:
                        continue
                    fs_for_other_label += d
                    inst_for_other_label += class_instance_count[l]

                if inst_for_other_label == 0:
                    score_sum_uniqueness += fs_for_label / inst_for_label
                else:
                    score_sum_uniqueness += fs_for_label / inst_for_label / (
                            fs_for_other_label / inst_for_other_label + 1)

            score_avg_uniqueness = score_sum_uniqueness / features

            csvwriter.writerow([label, score_avg_uniqueness])

    print("end")


# for each vocabulary entry in the dataset, calculate its distribution score found in each class
# (see method 'calc_dist_score')
def calc_distribution(input_data_file, sys_out, output_data_file, word_norm_option, label_col):
    raw_data = pd.read_csv(input_data_file, sep=',', encoding="utf-8")
    # call method to get vocabulary. need to change this method for n-grams
    M = cd.get_word_vocab(raw_data.tweet, sys_out, word_norm_option)
    # M=self.feature_scale(M)

    stats = {}
    all_label_inst = {}
    M0 = M[0]
    inverted_dict = dict([(v, k) for k, v in M[1].items()])
    for index, row in raw_data.iterrows():
        # print(index)
        vocab = M0[index]  # get the vocab for that tweet, only indices are returned
        label = row[label_col]  # get the label for that tweet
        if label in all_label_inst.keys():
            all_label_inst[label] += 1
        else:
            all_label_inst[label] = 1

        for v in vocab:
            str = inverted_dict[v]
            if str in stats.keys():
                dist = stats[str]
            else:
                dist = {}

            if label in dist.keys():
                dist[label] += 1
            else:
                dist[label] = 1
            stats[str] = dist

    # calc % of each class of instance
    sum_inst = 0
    for v in all_label_inst.values():
        sum_inst += v
    all_label_inst_perc = {}
    for k, v in all_label_inst.items():
        perc = v / sum_inst
        all_label_inst_perc[k] = perc

    with open(output_data_file, 'w', newline='\n') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                               quotechar='"', quoting=csv.QUOTE_MINIMAL)

        header = ["vocab"]
        all_labels = list(all_label_inst.keys())
        header = header + list(all_labels)
        csvwriter.writerow(header)
        for k, v in stats.items():
            row = [k]
            final_dist_scores = calc_dist_score(v, all_label_inst_perc)
            for l in all_labels:
                if l in final_dist_scores.keys():
                    row.append(final_dist_scores[l])
                else:
                    row.append("0")
            csvwriter.writerow(row)

    print("end")


def calc_dist_score(word_freq_ov_classes: dict, overall_label_inst_percentage: dict):
    sum = 0
    for freq_w in word_freq_ov_classes.values():
        sum += freq_w

    word_dist_score_ov_classes = {}
    sum_1 = 0
    for label_w, freq_w in word_freq_ov_classes.items():
        dist_score = freq_w / sum
        norm_1 = dist_score / overall_label_inst_percentage[label_w]
        word_dist_score_ov_classes[label_w] = norm_1
        sum_1 += norm_1

    final_word_dist_score_ov_classes = {}
    for l, s in word_dist_score_ov_classes.items():
        final_word_dist_score_ov_classes[l] = s / sum_1

    return final_word_dist_score_ov_classes


if __name__ == "__main__":
    # input_data="/home/zz/Work/chase/data/ml/ml/dt/labeled_data_all_2.csv"
    # sys_out="/home/zz/Work/chase/output"
    # output_data="/home/zz/Work/chase/output/word_dist_dt.csv"
    # word_norm_option=0
    # label_col=5
    # calc_distribution(input_data, sys_out,output_data,word_norm_option, label_col)
    # exit(0)

    # input_data = "/home/zz/Work/chase/data/ml/ml/rm/labeled_data_all.csv"
    # sys_out = "/home/zz/Work/chase/output"
    # output_data = "/home/zz/Work/chase/output/feature_uniqueness_rm.csv"
    # word_norm_option = 0
    # label_col = 6

    input_data = "/home/zz/Work/chase/data/ml/ml/ws-gb/labeled_data_all.csv"
    sys_out = "/home/zz/Work/chase/output"
    output_data = "/home/zz/Work/chase/output/feature_uniqueness_ws-gb.csv"
    word_norm_option = 0
    label_col = 6
    # calc_distribution(input_data, sys_out, output_data, word_norm_option, label_col)
    calc_instance_unique_feature_percent(input_data, sys_out, output_data, word_norm_option, label_col)
    # calc_average_feature_uniqueness(input_data, sys_out, output_data,word_norm_option, label_col)
