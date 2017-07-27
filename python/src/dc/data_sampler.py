import csv
import logging
import os
import random

from SolrClient import SolrClient


SOLR_SERVER="http://localhost:8983/solr"
SOLR_CORE="chase_searchapi"
#KEYWORDS='ban+kill+die+evil+hate+attack+terrorist+terrorism+threat+#DeportallMuslims+#refugeesnotwelcome'
KEYWORDS='*'


logger = logging.getLogger(__name__)
LOG_DIR=os.getcwd()+"/logs"
logging.basicConfig(filename=LOG_DIR+'/data_sampler.log', level=logging.INFO, filemode='w')

#query for particular date range to find out total
    #perform segmented query on given batch size k
    #randomly select n from each batch


def read_sampling_config(file):
    d = {}
    with open(file) as sampling_config_file:
        for line in sampling_config_file:
            name, var = line.partition("=")[::2]
            d[name]=var.strip()
    return d


def write_to_csv(file, list_of_tweets):
    with open(file, 'w', newline='',encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for tweet in list_of_tweets:
            writer.writerow(["",tweet['status_text'].replace("\n"," "), tweet['id']])


def query_solr(timespan, rows):
    solr = SolrClient(SOLR_SERVER)
    res = solr.query(SOLR_CORE,{
        'q':'status_text:'+KEYWORDS+' AND created_at:' + timespan,
        'rows':rows,
        'fl':'id,status_text'})

    rnd_res=res.docs
    random.shuffle(rnd_res)
    # final_res=[]
    # count=0
    # for doc in rnd_res:
    #     count+=1
    #     final_res.append(doc)
    #     if(count==sample_size):
    #         break
    return rnd_res


def collect_samples(config_file,out_folder):
    config = read_sampling_config(config_file)
    samples_per_timespan=int(config["samples_per_timespan"])
    elements_per_sample=int(config["elements_per_sample"])
    query_batch_size=samples_per_timespan*elements_per_sample*10
    timespans = config["timespans"].split(",")

    collected_sample_index=1

    for timespan_index in range(len(timespans)):
        logger.info("collecting for the #"+str(timespan_index)+" timespan")
        timespan = timespans[timespan_index]

        res=query_solr(timespan,query_batch_size)

        count_sample_elements=0
        count_samples=0
        sample=[]
        for doc in res:
            sample.append(doc)
            count_sample_elements+=1
            if count_sample_elements==elements_per_sample:
                count_samples+=1
                write_to_csv(out_folder+"/"+str(collected_sample_index)+".csv", sample)
                sample=[]
                count_sample_elements=0
                collected_sample_index+=1

            if count_samples==samples_per_timespan:
                break



# collect_samples("/home/zqz/Work/chase/config/data_sampling.txt",
#                 "/home/zqz/Work/chase/data/annotation/tag_filtered")
