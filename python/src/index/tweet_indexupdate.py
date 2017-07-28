import logging

import datetime

import sys
from SolrClient import SolrClient

from index import util as iu
from ml import util as mu
from ml.vectorizer import fv_chase_basic

logger = logging.getLogger(__name__)

def update_ml_tag(solr:SolrClient,
                  core_name,
                  docs, feat_vectorizer, ml_model, selected_features,
                  hate_indicative_features, scaling_option, sysout, logger):
    tweets=[]
    for d in docs:
        tweets.append(d['status_text'])

    #ml classify
    logger.info("begin ml classification for tweets={}, time={}".format(len(tweets), datetime.datetime.now()))
    tags = iu.ml_tag(tweets,feat_vectorizer, ml_model,selected_features,
                       hate_indicative_features,
                       scaling_option,sysout,logger)
    #compute risk score based on selected features and tag index


    logger.info("ml classification done. updating solr index...{}".format(datetime.datetime.now()))

    count=0
    for t, d in zip(tags,docs):
        if t==0:
            count+=1
            #print(d['status_text'])
        d['ml_tag']=str(t)
    print(count)
    solr.index(core_name, docs)
    code = iu.commit(core_name)


def update(solr:SolrClient,core_name,
           timespan, rows, feat_vectorizer, ml_model, selected_features,
           hate_indicative_features,
           scaling_option, sysout, logger):

    stop=False
    start=0
    while not stop:
        res = solr.query(core_name,{
            'q':'created_at:' + timespan,
            'rows':rows,
            'fl':'*',
            'start':start,
            'sort':'id asc'})
        start+=rows
        if start>res.num_found:
            stop=True

        #apply pretrained ML model to tag data and update them
        update_ml_tag(solr, core_name,
                      res.docs, feat_vectorizer, ml_model, selected_features,
                      hate_indicative_features,scaling_option, sysout, logger)
        #update tag index
        #update_hashtag_index(res.docs)
        rnd_res=res.docs
        #run machine learning to annotate data

        #update tag index

    pass


feat_vectorizer=fv_chase_basic.FeatureVectorizerChaseBasic()
ml_model=iu.load_ml_model(sys.argv[1])
selected_features = mu.read_preselected_features(False, sys.argv[2])
hate_indicative_features = mu.read_preselected_features(False, sys.argv[3])

solr = SolrClient(iu.solr_url)

update(solr,
       iu.solr_core_tweets,
       "[2017-07-27T01:00:00Z TO 2017-07-28T01:00:00Z]", 500,
       feat_vectorizer,ml_model,selected_features,
       hate_indicative_features,
       0,sys.argv[4],logger)
