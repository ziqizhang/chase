import urllib.request

import pickle

import datetime

from ml import util, text_preprocess
from ml import classifier_traintest as ct


solr_core_tweets="tweets"
solr_core_tags="tags"
solr_url="http://localhost:8983/solr"
tag_index_field_text="tag_text"
tag_index_field_type="type"
tag_index_field_frequency="frequency"
tag_index_field_frequencyh="frequencyh"
tag_index_field_pmi="pmi"
tag_index_field_risk_score="risk_score"
score_denominator_min=5

def commit(core_name):
    code = urllib.request. \
        urlopen("{}/{}/update?commit=true".
                format(solr_url,core_name)).read()

def ml_tag(tweets, feat_vectorizer, model, selected_features,
           hate_indicative_features,
           scaling,sysout, logger):
    logger.info("creating features and applying classification...")
    tweets_cleaned = [text_preprocess.preprocess_clean(x, True, True) for x in tweets]
    M = feat_vectorizer.transform_inputs(tweets, tweets_cleaned, sysout, "na")

    X_test_selected = ct.map_to_trainingfeatures(selected_features, M[1])
    X_test_selected = util.feature_scale(scaling, X_test_selected)
    labels = model.predict(X_test_selected)
    logger.info("computing hate risk scores based on hate indicative features... {}".format(datetime.datetime.now()))
    scores=compute_hate_riskscore(X_test_selected, M[1], hate_indicative_features)

    return labels, scores



def compute_hate_riskscore(X_, X_features_with_vocab, indicative_features: {}):

    scores = []

    # for each feature type in train data features
    for t_key, t_value in X_features_with_vocab.items():
        t_value_=t_value
        if isinstance(t_value_, set):
            t_value_=list(t_value)

        # if feature type exist in test data features
        if t_key in indicative_features:
            # for each feature in test data feature of this type
            fromdata_feature = indicative_features[t_key]
            fromdata_feature_vocab = fromdata_feature[1]
            fromdata_feature_value = fromdata_feature[0]
            # for each data instance
            from_features_row_index = 0
            row_index=0
            for row in fromdata_feature_value:
                if filter is not None and from_features_row_index not in filter:
                    from_features_row_index += 1
                    continue

                for vocab, value in zip(fromdata_feature_vocab, row):  # feature-value pair for each instance in test
                    # check if that feature exists in train data features of that type
                    if vocab in t_value_:
                        # if so, se corresponding feature value in the new feature vector
                        if isinstance(t_value_, dict):
                            vocab_index = t_value_[vocab]
                        else:
                            vocab_index = t_value_.index(vocab)

    return scores



def load_ml_model(file):
    with open(file, 'rb') as model:
        return pickle.load(model)
