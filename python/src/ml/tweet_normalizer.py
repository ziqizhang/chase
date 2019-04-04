import csv
import re

import pandas as pd
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

text_processor = TextPreProcessor(
    # terms that will be normalized
# normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
#                'time', 'url', 'date', 'number'],
#     # terms that will be annotated
#     annotate={"hashtag", "allcaps", "elongated", "repeated",
#               'emphasis', 'censored'},

    normalize=[],
    # terms that will be annotated
    annotate={'elongated',
              'emphasis'},
    fix_html=True,  # fix HTML tokens

    # corpus from which the word statistics are going to be used
    # for word segmentation
    segmenter="twitter",

    # corpus from which the word statistics are going to be used
    # for spell correction
    corrector="twitter",

    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words

    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=False).tokenize,

    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)


sentences = [
    "#YouTube",
    "CANT WAIT for the new season of #TwinPeaks ＼(^o^)／!!! #davidlynch #tvseries :)))",
    "I saw the new #johndoe movie and it suuuuucks!!! WAISTED $10... #badmovies :/",
    "@SentimentSymp:  hoooolly can't wait for the Nov 9 #Sentiment talks! *VERY* good, f**k YAAAAAAY !!! :-D http://sentimentsymposium.com/.",
    "Add anotherJEW fined a bi$$ion for stealing like a lil maggot"
]

for s in sentences:
    res=text_processor.pre_process_doc(s)
    res=list(filter(lambda a: a != '<elongated>', res))
    print(res)
exit(1)

col_text=7
input_data_file="/home/zz/Work/chase/data/ml/ml/rm/labeled_data_all_corrected.csv"
#input_data_file="/home/zz/Work/chase/data/ml/ml/dt/labeled_data_all_2.csv"
#input_data_file="/home/zz/Work/chase/data/ml/ml/w/labeled_data_all.csv"
#input_data_file="/home/zz/Work/chase/data/ml/ml/w+ws/labeled_data_all.csv"
#input_data_file="/home/zz/Work/chase/data/ml/ml/ws-exp/labeled_data_all.csv"
#input_data_file="/home/zz/Work/chase/data/ml/ml/ws-amt/labeled_data_all.csv"
#input_data_file="/home/zz/Work/chase/data/ml/ml/ws-gb/labeled_data_all.csv"

raw_data = pd.read_csv(input_data_file, sep=',', encoding="utf-8")
header_row=list(raw_data.columns.values)
with open(input_data_file+"c.csv", 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csvwriter.writerow(header_row)

    for row in raw_data.iterrows():
        tweet = list(row[1])
        tweet_text=text_processor.pre_process_doc(tweet[col_text])
        tweet_text = list(filter(lambda a: a != '<elongated>', tweet_text))
        tweet_text = list(filter(lambda a: a != '<emphasis>', tweet_text))
        tweet_text = list(filter(lambda a: a != 'RT', tweet_text))
        tweet_text = list(filter(lambda a: a != '"', tweet_text))
        tweet_text=" ".join(tweet_text)

        #reset content
        tweet[col_text]=tweet_text

        csvwriter.writerow(tweet)
