import logging

import numpy

from ml import feature_extractor
from ml import util, text_preprocess
from ml import classifier_traintest as ct
import datetime

import sys
from SolrClient import SolrClient

from index import util as iu
from ml import util as mu
from ml.vectorizer import fv_chase_basic

logger = logging.getLogger(__name__)

def ml_tag(tweets, feat_vectorizer, model, selected_features,
           hate_indicative_features,
           scaling,sysout, logger,solr_tags:SolrClient, core_name):
    logger.info("creating features and applying classification...{}".format(datetime.datetime.now()))
    tweets_cleaned = [text_preprocess.preprocess_clean(x, True, True) for x in tweets]
    M = feat_vectorizer.transform_inputs(tweets, tweets_cleaned, sysout, "na")

    X_test_selected = ct.map_to_trainingfeatures(selected_features, M[1])
    X_test_selected = util.feature_scale(scaling, X_test_selected)
    labels = model.predict(X_test_selected)
    logger.info("computing hate risk scores... {}".format(datetime.datetime.now()))
    scores=compute_hate_riskscore_by_features(X_test_selected, M[1], hate_indicative_features, solr_tags, core_name)

    return labels, scores



def compute_hate_riskscore_by_features(X_, X_features_with_vocab, indicative_features: {}, solr_tags:SolrClient, core_name):
    scores = []
    X_features_with_vocab_flattened={}
    for k, v in X_features_with_vocab.items():
        values=v[1]
        if isinstance(values, dict):
            values = {v: k for k, v in v[1].items()}
        X_features_with_vocab_flattened[k]=(v[0], values)


    for i in range(0, len(X_)):
        nonzero_feat=numpy.count_nonzero(X_[i])
        indicative_feat=0
        tag_modifier=0
        tag_count=0
        for if_k, if_v in indicative_features.items():
            if if_k in X_features_with_vocab_flattened.keys():
                feats_of_type=X_features_with_vocab_flattened[if_k]
                feats_of_instance=feats_of_type[0][i]
                feats_of_instance_nonzero=[i for i, e in enumerate(feats_of_instance) if e != 0]

                for f in feats_of_instance_nonzero:
                    fv=feats_of_type[1][f]
                    if fv in if_v:
                        indicative_feat+=1
                    if if_k == feature_extractor.TWEET_HASHTAG_FEATURES_VOCAB:
                        tag_modifier+=get_tag_riskscore(solr_tags, core_name, fv)
                        tag_count+=1
                        pass
        if tag_count>0:
            tag_modifier=tag_modifier/tag_count

        if_ratio = numpy.math.sqrt(indicative_feat/nonzero_feat)*(1+tag_modifier)
        if nonzero_feat==0:
            if_ratio=0
        scores.append(if_ratio)

    return scores


def get_tag_riskscore(solr:SolrClient, core_name, tag):
    if tag[0]=='#':
        tag=tag[1:]
    tag=tag.lower()
    res = solr.query(core_name, {
            'q': 'id:'+tag,
            'fl': iu.tag_index_field_risk_score})
    for d in res.docs:
        score= d[iu.tag_index_field_risk_score]
        return score
    return 0.0


def update_ml_tag(solr:SolrClient,
                  tweets_core_name,
                  tags_core_name,
                  docs, feat_vectorizer, ml_model, selected_features,
                  hate_indicative_features, scaling_option, sysout, logger):
    tweets=[]
    for d in docs:
        text=d['status_text']
        if "rt @" in text.lower():
            start=text.lower().index("rt @")+4
            text=text[start].strip()

        tweets.append(text)

    #ml classify, also compute risk scores
    logger.info("begin ml classification for tweets={}, time={}".format(len(tweets), datetime.datetime.now()))
    tags, risk_scores = ml_tag(tweets,feat_vectorizer, ml_model,selected_features,
                       hate_indicative_features,
                       scaling_option,sysout,logger, solr, tags_core_name)

    logger.info("ml classification done. updating solr index...{}".format(datetime.datetime.now()))

    count=0
    for idx, tag in enumerate(tags):
        if tag==0:
            count+=1
            #print(d['status_text'])
        d = docs[idx]
        d['ml_tag']=str(tag)
        d['tweet_risk']=risk_scores[idx]

    print(count)
    solr.index(tweets_core_name, docs)
    code = iu.commit(tweets_core_name)


def update(solr:SolrClient, tweet_core_name, tag_core_name,
           timespan, rows, feat_vectorizer, ml_model, selected_features,
           hate_indicative_features,
           scaling_option, sysout, logger):

    stop=False
    start=0
    while not stop:
        logger.warn("Processing from {} for a batch of {}".format(start, rows))
        print("Processing from {} for a batch of {}".format(start, rows))
        res = solr.query(tweet_core_name, {
            'q':'created_at:' + timespan,
            'rows':rows,
            'fl':'*',
            'start':start,
            'sort':'id asc'})
        start+=rows
        if start>res.num_found:
            stop=True

        #apply pretrained ML model to tag data and update them
        update_ml_tag(solr, tweet_core_name, tag_core_name,
                      res.docs, feat_vectorizer, ml_model, selected_features,
                      hate_indicative_features, scaling_option, sysout, logger)

    pass


feat_vectorizer=fv_chase_basic.FeatureVectorizerChaseBasic()
ml_model=iu.load_ml_model(sys.argv[1])
selected_features = mu.read_preselected_features(False, sys.argv[2])
hate_indicative_features = mu.read_preselected_features(False, sys.argv[3])

solr = SolrClient(iu.solr_url)

logging.basicConfig(level=logging.INFO)
#todo: how should we deal with retweet? currently it is treated as a new tweet and we simply remove the 'rt @' from text

update(solr,
       iu.solr_core_tweets,
       iu.solr_core_tags,
       "[2017-07-29T01:00:00Z TO 2017-08-08T01:00:00Z]", 1000,
       feat_vectorizer,ml_model,selected_features,
       hate_indicative_features,
       0,sys.argv[4],logger)
