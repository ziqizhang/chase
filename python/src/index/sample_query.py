'''
Firstly start the server by:
$ cd solr-6.6.0/bin
$ ./solr start -s [/home/.../chase/data/solr]


Tips for using solr server (https://cwiki.apache.org/confluence/display/solr/Running+Solr)
- always remember TO STOP THE SERVER when you finish, by typing './solr stop -all'
- it is better to make a back up of the index. this is located in [/home/.../chase/data/solr]
- sometimes if server does not start, check if there is a 'write.lock' file within the
sub folders of [/home/.../chase/data/solr]. if so, delete them and try again


solr query syntax: https://wiki.apache.org/solr/SolrQuerySyntax
and lots of other resources by goolging...

'''

from SolrClient import SolrClient
from index import util as iu

#get results matching a time interval, and apply pagination
#the core_name should match [/home/.../chase/data/solr/FOLDERNAME]
def get_tweets_by_time(timespan, solr:SolrClient, core_name="tweets"):
    rows=100 #100 results per page
    stop=False
    start=0
    facet_counts=None

    q='created_at:' + timespan+' AND ml_tag:0'

    while not stop:
        res = solr.query(core_name, {
            'q':q, #remember we only show tweets tagged as hate (0)
            'facet.field':'entities_hashtag', #count results per facet (NOTE, not every tweet will have a hashtag, but this is ok
            'facet':"on", #switch on facet search
            'facet.mincount':"1", #show facets that have at least 1 result
            'rows':rows,
            'fl':'*',   #return all fields from the index (when available
            'start':start, #start from
            'sort':'tweet_risk desc'}) #sort by risk_score descending
        start+=rows #resetting start will turn to next page. for specific page number, you need to work out the 'start' by pagenum*rows
        print("total number found={}".format(res.num_found))
        if start>res.num_found:
            stop=True

        #assign facet results to another var. facet counts is for the whole dataset, not just this page
        if facet_counts is None:
            facet_counts=res.data['facet_counts']['facet_fields']['entities_hashtag']

        #now go through every page, every result
        for d in res.docs: #res.docs only contain documents on the CURRENT page
            print("https://twitter.com/"+d['user_screen_name']+"/"+d['id'])
            if 'coordinates' in d.keys():
                print(d['coordinates'])

    #finally print facet counts
    print(facet_counts)


#based on get_tweets_by_time but also apply a tag match
#because tag is a multi-valued field, we must amend our query but adding a 'facet.query' parameter,
#but not to the 'q' parameter
def get_tweets_by_time_and_tag(tag, timespan, solr:SolrClient, core_name="tweets"):
    rows=100 #100 results per page
    stop=False
    start=0
    facet_counts=None
    res = solr.query(core_name, {
            'q':'created_at:' + timespan+' AND ml_tag:0'
                    +' AND entities_hashtag:'+tag, #remember we only show tweets tagged as hate (0)
            'facet.field':'entities_hashtag', #count results per facet (NOTE, not every tweet will have a hashtag, but this is ok
            'facet':"on", #switch on facet search
            'facet.mincount':"1", #show facets that have at least 1 result
            'rows':rows,
            'fl':'*',   #return all fields from the index (when available
            'start':start, #start from
            'sort':'tweet_risk desc'}) #sort by risk_score descending
    #the rest code should be the same as get_tweets_by_time_and_tag


def get_tweets_by_coordinates(lat, lon, range, timespan, solr:SolrClient, core_name="tweets"):
    lat_min=lat-range
    lat_max=lat+range
    lon_min=lon-range
    lon_max=lon+range
    rows=100 #100 results per page
    stop=False
    start=0
    facet_counts=None
    res = solr.query(core_name, {
            'q':'created_at:' + timespan+' AND ml_tag:0'
                    +' AND coordinate_lat:'+'[{} TO {}]'.format(lat_min, lat_max)
                    +' AND coordinate_lon:'+'[{} TO {}]'.format(lon_min, lon_max),
            'facet.field':'entities_hashtag', #count results per facet (NOTE, not every tweet will have a hashtag, but this is ok
            'facet':"on", #switch on facet search
            'facet.mincount':"1", #show facets that have at least 1 result
            'rows':rows,
            'fl':'*',   #return all fields from the index (when available
            'start':start, #start from
            'sort':'tweet_risk desc'}) #sort by risk_score descending
    #the rest code should be the same as get_tweets_by_time_and_tag


#given a query tag, get top N related ranked by pmi
def get_tags_by_pmi(target_tag, solr:SolrClient, core_name="tags"):
    #http://localhost:8983/solr/tags/select?indent=on&q=tag_text:banmuslims%20AND%20type:1&wt=json
    rows=100 #100 results per page
    stop=False
    start=0

    q='tag_text:' + target_tag+' AND type:1' #0=single tag; 1=tag pairs
    #because we need to get tags similar to this target, so we need to get all pairs and process them

    while not stop:
        res = solr.query(core_name, {
            'q':q, #remember we only show tweets tagged as hate (0)
            'rows':rows,
            'fl':'*',   #return all fields from the index (when available
            'start':start, #start from
            'sort':'pmi desc'}) #sort by risk_score descending
        start+=rows #resetting start will turn to next page. for specific page number, you need to work out the 'start' by pagenum*rows
        print("total number found={}".format(res.num_found))
        if start>res.num_found:
            stop=True

        #now go through every page, every result
        for d in res.docs: #res.docs only contain documents on the CURRENT page
            tags=d['tag_text'].split(" ")
            relevant_tag=tags[0]
            if relevant_tag==target_tag:
                relevant_tag=tags[1]
            print(relevant_tag+", pmi="+d['pmi'])



###################
solr=SolrClient(iu.solr_url)
timespan="[2017-07-29T22:40:00Z TO 2017-07-29T22:45:00Z]"
#notice how solr formats time. we must use the exact format
get_tweets_by_time(timespan, solr)
#the following method is incomplete, just an example. step in to see comments
get_tweets_by_time_and_tag('banmuslim', timespan,
                           solr)
#get tweets within a region. given a coordinate point, we want to draw a range around that point
#so we need to use a ranged query
get_tweets_by_coordinates(52.8,-0.5,1,timespan, solr)

#get a list of tags that co-occur with this one and print their scores
get_tags_by_pmi('banmuslims',solr)


