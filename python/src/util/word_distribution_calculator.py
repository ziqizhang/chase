import csv

from ml import classifier_dnn as cd
import pandas as pd


def calc_distribution(input_data_file, sys_out, output_data_file, word_norm_option, label_col):
    raw_data = pd.read_csv(input_data_file, sep=',', encoding="utf-8")
    M = cd.get_word_vocab(raw_data.tweet, sys_out, word_norm_option)
    # M=self.feature_scale(M)

    stats={}
    all_label_inst={}
    M0 = M[0]
    inverted_dict=dict([ (v, k) for k, v in M[1].items()])
    for index, row in raw_data.iterrows():
        #print(index)
        vocab=M0[index] #vocab in that tweet as indices
        label=row[label_col] #label for that tweet
        if label in all_label_inst.keys():
            all_label_inst[label]+=1
        else:
            all_label_inst[label]=1

        for v in vocab:
            str=inverted_dict[v]
            if str in stats.keys():
                dist=stats[str]
            else:
                dist={}

            if label in dist.keys():
                dist[label]+=1
            else:
                dist[label]=1
            stats[str]=dist

    #calc % of each class of instance
    sum_inst=0
    for v in all_label_inst.values():
        sum_inst+=v
    all_label_inst_perc={}
    for k,v in all_label_inst.items():
        perc=v/sum_inst
        all_label_inst_perc[k]=perc

    with open(output_data_file, 'w', newline='\n') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)

        header=["vocab"]
        all_labels = list(all_label_inst.keys())
        header=header + list(all_labels)
        csvwriter.writerow(header)
        for k, v in stats.items():
            row=[k]
            final_dist_scores=calc_dist_score(v, all_label_inst_perc)
            for l in all_labels:
                if l in final_dist_scores.keys():
                    row.append(final_dist_scores[l])
                else:
                    row.append("0")
            csvwriter.writerow(row)

    print("end")


def calc_dist_score(word_freq_ov_classes:dict, overall_label_inst_percentage:dict):
    sum=0
    for freq_w in word_freq_ov_classes.values():
        sum+=freq_w

    word_dist_score_ov_classes={}
    sum_1=0
    for label_w, freq_w in word_freq_ov_classes.items():
        dist_score=freq_w/sum
        norm_1=dist_score/overall_label_inst_percentage[label_w]
        word_dist_score_ov_classes[label_w]=norm_1
        sum_1+=norm_1

    final_word_dist_score_ov_classes={}
    for l, s in word_dist_score_ov_classes.items():
        final_word_dist_score_ov_classes[l]=s/sum_1

    return final_word_dist_score_ov_classes


if __name__ == "__main__":
    # input_data="/home/zz/Work/chase/data/ml/ml/dt/labeled_data_all_2.csv"
    # sys_out="/home/zz/Work/chase/output"
    # output_data="/home/zz/Work/chase/output/word_dist_dt.csv"
    # word_norm_option=0
    # label_col=5
    # calc_distribution(input_data, sys_out,output_data,word_norm_option, label_col)
    input_data = "/home/zz/Work/chase/data/ml/ml/w/labeled_data_all.csv"
    sys_out = "/home/zz/Work/chase/output"
    output_data = "/home/zz/Work/chase/output/word_dist_w.csv"
    word_norm_option = 0
    label_col = 6
    calc_distribution(input_data, sys_out, output_data, word_norm_option, label_col)

