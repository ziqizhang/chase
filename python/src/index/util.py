import urllib.request

import pickle




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


def load_ml_model(file):
    with open(file, 'rb') as model:
        return pickle.load(model)
