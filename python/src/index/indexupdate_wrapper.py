

'''order of update:
for every [time_interval]
1. tag_indexupdate - update all tag scores, this requires a list of tags for a list of tweets. Where those tweets come from
depends on individual choices
2. tweet_indexupdate - classify all tweets; compute tweet risk score using tag_index
'''
