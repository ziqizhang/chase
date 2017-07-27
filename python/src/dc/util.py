import csv
import os

import pickle

from ml import util, text_preprocess
from ml import classifier_traintest as ct

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

def ml_tag(tweets, feat_vectorizer, model, selected_features, scaling,sysout, logger):
    tweets_cleaned = [text_preprocess.preprocess_clean(x, True, True) for x in tweets]
    M = feat_vectorizer.transform_inputs(tweets, tweets_cleaned, sysout, "na")      

    X_test_selected = ct.map_to_trainingfeatures(selected_features, M[1])
    X_test_selected = util.feature_scale(scaling, X_test_selected)
    labels = model.predict(X_test_selected)
    return labels


def load_ml_model(file):
    with open(file, 'rb') as model:
        return pickle.load(model)

# in_folder="/home/zqz/Work/chase/data/annotation/unfiltered"
# out_file="/home/zqz/Work/chase/data/annotation/unfilered_merged.csv"
# in_folder="/home/zqz/Work/chase/data/annotation/keyword_filtered"
# out_file="/home/zqz/Work/chase/data/annotation/keywordfilered_merged.csv"
# print(os.getcwd())
# in_folder="../../../data/annotation/tag_filtered"
# out_file="../../../data/annotation/tagfilered_merged.csv"
# merge_annotations(in_folder,out_file)
