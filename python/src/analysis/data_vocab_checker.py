import functools
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from ml import nlp
from ml import text_preprocess as tp


def get_word_vocab(tweets, normalize_option):
    word_vectorizer = CountVectorizer(
        # vectorizer = sklearn.feature_extraction.text.CountVectorizer(
        tokenizer=functools.partial(nlp.tokenize, stem_or_lemma=normalize_option),
        preprocessor=tp.strip_hashtags,
        ngram_range=(1, 1),
        stop_words=nlp.stopwords,  # We do better when we keep stopwords
        decode_error='replace',
        max_features=50000,
        min_df=1,
        max_df=0.999
    )

   # logger.info("\tgenerating word vectors, {}".format(datetime.datetime.now()))
    counts = word_vectorizer.fit_transform(tweets).toarray()
    #logger.info("\t\t complete, dim={}, {}".format(counts.shape, datetime.datetime.now()))
    vocab = {v: i for i, v in enumerate(word_vectorizer.get_feature_names())}
    #pickle.dump(vocab, open(out_folder + "/" + "DNN_WORD_EMBEDDING" + ".pk", "wb"))

    word_embedding_input = []
    for tweet in counts:
        tweet_vocab = []
        for i in range(0, len(tweet)):
            if tweet[i] != 0:
                tweet_vocab.append(i)
        word_embedding_input.append(tweet_vocab)
    return word_embedding_input, vocab

#calculate I2U and U2C stats
def check_stats(processed_tweets, vocab, raw_data):
    vocab_by_class={}
    instance_count={}
    for i in range(0, len(processed_tweets)):
        tweet=processed_tweets[i]
        raw_tweet=raw_data[i]
        gs_label=raw_tweet[6]

        if gs_label in vocab_by_class.keys():
            voc=vocab_by_class[gs_label]
            voc.update(tweet)
        else:
            voc=set()
            voc.update(tweet)
        vocab_by_class[gs_label]=voc

        if gs_label in instance_count.keys():
            instance_count[gs_label]+=1
        else:
            instance_count[gs_label]=1

    print("total,"+str(len(vocab)))
    print("I2U stats:")
    for k, v in vocab_by_class.items():
        print(str(k)+","+str(len(v))+","+str(instance_count[k])+","+str(instance_count[k]/len(v)))

    keys=list(vocab_by_class.keys())
    ##### calc intersection ####
    # for i in range(0, len(keys)):
    #     first=keys[i]
    #     for j in range(i+1, len(keys)):
    #         second=keys[j]
    #
    #         first_v=vocab_by_class[first]
    #         second_v=vocab_by_class[second]
    #
    #         inter=first_v&second_v
    #         print(str(first)+" and "+str(second)+" has common "+str(len(inter)))

    ##### calc diff ####
    print("U2C stats")
    for i in range(0, len(keys)):
        first=keys[i]
        first_v=vocab_by_class[first]
        second_v=set()
        for j in range(0, len(keys)):
            if i==j:
                continue
            second=keys[j]
            second_v.update(vocab_by_class[second])
        diff=first_v-second_v
        print(str(first)+" vs others has diff "+str(len(diff))+","+str(len(diff)/len(first_v)))


def check_stats_of_classes(class1, class2, processed_tweets, vocab, raw_data):
    vocab_dist={}

    for i in range(0, len(processed_tweets)):
        tweet = processed_tweets[i]
        raw_tweet = raw_data[i]
        gs_label = raw_tweet[6]
        if gs_label in vocab_dist.keys():
            vocab_dist[gs_label].update(tweet)
        else:
            tokens=set()
            vocab_dist[gs_label]=tokens

    vocab_of_class1 = vocab_dist[class1]
    vocab_of_class2 = vocab_dist[class2]

    all_but_class1=set()
    all_but_class2=set()
    for k, v in vocab_dist.items():
        if k==class1:
            continue
        all_but_class1.update(v)
    for k, v in vocab_dist.items():
        if k==class2:
            continue
        all_but_class2.update(v)


    class1_unique=vocab_of_class1.difference(all_but_class1)
    class2_unique=vocab_of_class2.difference(all_but_class2)

    non_unique_of_vocab_of_class1=vocab_of_class1.difference(class2_unique)
    non_unique_of_vocab_of_class2=vocab_of_class2.difference(class1_unique)

    ##### calc intersection ####
    # for i in range(0, len(keys)):
    #     first=keys[i]
    #     for j in range(i+1, len(keys)):
    #         second=keys[j]
    #
    #         first_v=vocab_by_class[first]
    #         second_v=vocab_by_class[second]
    #
    #         inter=first_v&second_v
    #         print(str(first)+" and "+str(second)+" has common "+str(len(inter)))

    ##### calc diff ####
    inter=set.intersection(non_unique_of_vocab_of_class1,non_unique_of_vocab_of_class2)
    print("label {} has {} unique, {} nonunique, intersection of non unique,{}".
          format(class1, len(class1_unique), len(non_unique_of_vocab_of_class1), len(inter)))
    print("label {} has {} unique, {} nonunique, intersection of non unique,{}".
          format(class2, len(class2_unique), len(non_unique_of_vocab_of_class2), len(inter)))

#input_data_file="/home/zz/Work/chase/data/ml/ml/rm/labeled_data_all.csv"
#input_data_file="/home/zz/Work/chase/data/ml/ml/dt/labeled_data_all_2.csv"
input_data_file="/home/zz/Work/chase/data/ml/ml/w/labeled_data_all.csv"
#input_data_file="/home/zz/Work/chase/data/ml/ml/w+ws/labeled_data_all.csv"
#input_data_file="/home/zz/Work/chase/data/ml/ml/ws-exp/labeled_data_all.csv"
#input_data_file="/home/zz/Work/chase/data/ml/ml/ws-amt/labeled_data_all.csv"
#input_data_file="/home/zz/Work/chase/data/ml/ml/ws-gb/labeled_data_all.csv"
raw_data = pd.read_csv(input_data_file, sep=',', encoding="utf-8")
M = get_word_vocab(raw_data.tweet, 0)
#check_stats(M[0], M[1], raw_data.as_matrix())
check_stats_of_classes(0, 1,M[0], M[1], raw_data.as_matrix())





