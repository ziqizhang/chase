from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from analysis import embedding_vocab_checker as evc
import pandas as pd

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

model = evc.load_model("/home/zz/Work/data/GoogleNews-vectors-negative300.bin.gz")
#model = evc.load_model("/home/zz/Work/data/Set1_TweetDataWithoutSpam_Word.bin")
#model = evc.load_model("/home/zz/Work/data/glove.840B.300d.bin.gensim")
raw_data = pd.read_csv("/home/zz/Work/chase/data/ml/ml/rm/labeled_data_all.csv",
                       sep=',', encoding="utf-8")
col_text=7

hashtags=0
included_tags=0
hashtag_toks=0
included_hashtag_toks=0

count_row=0
for row in raw_data.iterrows():
    count_row+=1
    print(count_row)
    tweet = list(row[1])[col_text]
    if '#' in tweet:
        toks = tweet.split(" ")
        for t in toks:
            if t.startswith("#"):
                hashtags+=1
                if t in model.wv.vocab.keys():
                    included_tags+=1
                norms=text_processor.pre_process_doc(t)
                hashtag_toks+=len(norms)
                for n in norms:
                    if len(n)<2:
                        continue
                    if n.lower() in model.wv.vocab.keys():
                        included_hashtag_toks+=1
print(str(hashtags)+","+str(included_tags)+","+str(hashtag_toks)+","+str(included_hashtag_toks))

