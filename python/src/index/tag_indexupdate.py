import logging

import numpy
import pandas as pd
import sys
from SolrClient import SolrClient
from ml import feature_extractor as fe

# get data about tags in existing tag index
from index import util

logger = logging.getLogger(__name__)


def get_existing(solr: SolrClient, core_name, pagesize):
    stop = False
    start = 0
    tags = {}
    tag_pairs = {}
    while not stop:
        res = solr.query(core_name, {
            'q': '*:*',
            'rows': pagesize,
            'fl': '*',
            'start': start})
        start += pagesize
        if start > res.num_found:
            stop = True

        for d in res.docs:
            if d['type'] == '0':  # single tag
                tags[d['id']] = d
            else:
                tag_pairs[d['id']] = d
    return tags, tag_pairs


def generate_pairs(tags: list):
    tags = sorted(tags)
    index = 1
    pairs = []
    for element1 in tags:
        for element2 in tags[index:]:
            pairs.append(element1 + " " + element2)
        index += 1

    return pairs


def update_frequencies(existing_tags: dict,
                       existing_tag_pairs: dict,
                       tags_of_tweets: list, tweet_labels: list, valid_label):
    count = 0
    for tags, label in zip(tags_of_tweets, tweet_labels):
        count += 1
        if (count % 100 == 0):
            logger.info("\t processed={}".format(count))
        # update single tag data
        for tag in tags:
            freq = 1
            freqh = 0
            if label == valid_label:
                freqh = 1

            tag_record = {"id": tag}
            if tag in existing_tags.keys():
                tag_record = existing_tags[tag]
                e_freq = int(tag_record[util.tag_index_field_frequency])
                freq += e_freq
                e_freqh = int(tag_record[util.tag_index_field_frequencyh])
                freqh += e_freqh
            tag_record[util.tag_index_field_frequency] = freq
            tag_record[util.tag_index_field_frequencyh] = freqh
            existing_tags[tag] = tag_record

        if label == valid_label:
            # update tag pair data
            pairs = generate_pairs(tags)
            for pair in pairs:
                freq = 1
                pair_record = {"id": pair}
                if pair in existing_tag_pairs.keys():
                    pair_record = existing_tag_pairs[pair]
                    e_freq = int(pair_record[util.tag_index_field_frequency])
                    freq += e_freq
                pair_record[util.tag_index_field_frequency] = freq
                existing_tag_pairs[pair] = pair_record


def update_pmi_scores(existing_tags: dict,
                      existing_tag_pairs: dict,
                      solr: SolrClient, core_name, batch_commit):
    count = 0
    batch = []
    for tag_pair, data in existing_tag_pairs.items():
        count += 1
        if count > batch_commit:
            solr.index(core_name, batch)
            code = util.commit(core_name)
            count = 0
            batch = []
            logger.info("\t done batch size={}".format(batch_commit))

        co_freq = data[util.tag_index_field_frequency]
        tags = tag_pair.split(" ")
        t1_freq = existing_tags[tags[0]][util.tag_index_field_frequency]
        t2_freq = existing_tags[tags[1]][util.tag_index_field_frequency]

        if co_freq==0:
            pmi=0
        else:
            pmi = numpy.emath.log(co_freq / (t1_freq * t2_freq + util.score_denominator_min))
        data[util.tag_index_field_pmi] = pmi
        data[util.tag_index_field_text] =tag_pair
        data[util.tag_index_field_type] =1
        batch.append(data)

    # commit the rest
    solr.index(core_name, batch)
    code = util.commit(core_name)


# risk score of a single tag is computed by log(freq(tag) in hate tweets / freq(tag) in all tweets)
def update_tagrisk_scores(existing_tags: dict,
                          solr: SolrClient, core_name, batch_commit):
    count = 0
    batch = []
    for tag, data in existing_tags.items():
        count += 1
        if count > batch_commit:
            solr.index(core_name, batch)
            code = util.commit(core_name)
            count = 0
            batch = []
            logger.info("\t done batch size={}".format(batch_commit))

        freq = data[util.tag_index_field_frequency]
        freqh = data[util.tag_index_field_frequencyh]

        if freqh==0:
            riskscore=0
        else:
            riskscore = numpy.math.sqrt(freqh / (freq+ util.score_denominator_min))
        data[util.tag_index_field_risk_score] = riskscore
        data[util.tag_index_field_text] =tag
        data[util.tag_index_field_type] =0
        batch.append(data)

    # commit the rest
    solr.index(core_name, batch)
    code = util.commit(core_name)


def recompute_tags(existing_tags: dict,
                   existing_tag_pairs: dict,
                   tags_of_tweets: list, tweet_labels: list, valid_label,
                   solr: SolrClient, core_name, batch_commit):
    logger.info("updating tag pair and single tag frequency data for tweets={}"
                " with existing tags={}, tag pairs={}".format(len(tweet_labels),
                                                              len(existing_tags), len(existing_tag_pairs)))
    update_frequencies(existing_tags,
                       existing_tag_pairs,
                       tags_of_tweets, tweet_labels, valid_label)

    logger.info("update complete, re-compute tag_pair pmi scores")
    update_pmi_scores(existing_tags, existing_tag_pairs, solr, core_name, batch_commit)
    logger.info("complete, re-compute tag risk scores")
    update_tagrisk_scores(existing_tags, solr, core_name, batch_commit)
    logger.info("complete")


def populate_index_coldstart(csv_data_file, solr: SolrClient, core_name, pagesize, batch_commit, sysout):
    existing_tags, existing_tag_pairs = get_existing(solr, core_name, pagesize)

    raw_data = pd.read_csv(csv_data_file, sep=',', encoding="utf-8")
    labels = raw_data['class']
    m, vocab = fe.get_hashtags_in_tweets(raw_data.tweet, sysout)
    tags=[]
    vocab_inv = {v: k for k, v in vocab.items()}
    for tweet in m:
        tgs =[]
        for index in [i for i, e in enumerate(tweet) if e != 0]:
            tgs.append(vocab_inv[index][1:])
        tags.append(tgs)

    recompute_tags(existing_tags, existing_tag_pairs, tags, labels, 0, solr, core_name, batch_commit)


##############################
solr = SolrClient(util.solr_url)

populate_index_coldstart(sys.argv[1],
                         solr,
                         util.solr_core_tags,
                         5000,  # pagesize
                         500,  # batch to commit
                         sys.argv[2])
