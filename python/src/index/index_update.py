import logging

import datetime

import sys
from SolrClient import SolrClient
import urllib.request

from dc import util
from ml import util as mu
from ml.vectorizer import fv_chase_basic

SOLR_CORE_TWEET="tweets"
solr = SolrClient("http://localhost:8983/solr")
SOLR_CORE_TAG="tags"

logger = logging.getLogger(__name__)

def update_ml_tag(docs, feat_vectorizer, ml_model, selected_features, scaling_option, sysout, logger):
    tweets=[]
    for d in docs:
        tweets.append(d['status_text'])
    tags = util.ml_tag(tweets,feat_vectorizer, ml_model,selected_features,scaling_option,sysout,logger)
    #update
    logger.info("ml classification done. updating solr index...{}".format(datetime.datetime.now()))

    count=0
    for t, d in zip(tags,docs):
        if t==0:
            count+=1
            #print(d['status_text'])
        d['ml_tag']=str(t)
    print(count)
    solr.index(SOLR_CORE_TWEET, docs)
    code = urllib.request. \
        urlopen("http://localhost:8983/solr/{}/update?commit=true".
                format(SOLR_CORE_TWEET)).read()


def update(timespan, rows, feat_vectorizer, ml_model, selected_features, scaling_option, sysout, logger):

    stop=False
    start=0
    while not stop:
        res = solr.query(SOLR_CORE_TWEET,{
            'q':'created_at:' + timespan,
            'rows':rows,
            'fl':'*',
            'start':start,
            'sort':'id asc'})
        start+=rows
        if start>res.num_found:
            stop=True

        #apply pretrained ML model to tag data and update them
        update_ml_tag(res.docs, feat_vectorizer, ml_model, selected_features, scaling_option, sysout, logger)
        #update tag index
        #update_hashtag_index(res.docs)
        rnd_res=res.docs
        #run machine learning to annotate data

        #update tag index

    pass


feat_vectorizer=fv_chase_basic.FeatureVectorizerChaseBasic()
ml_model=util.load_ml_model(sys.argv[1])
selected_features = mu.read_preselected_features(False, sys.argv[2])

update("[2017-07-27T01:00:00Z TO 2017-07-28T01:00:00Z]", 500, feat_vectorizer,ml_model,selected_features,0,sys.argv[3],logger)
