import re
import enchant
import splitter


#This is the original preprocess method from Davidson
def preprocess(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    emoji_regex = '&#[0-9]{4,6};'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, '', parsed_text)
    parsed_text = re.sub(mention_regex, '', parsed_text)
    parsed_text = re.sub('RT','', parsed_text) #Some RTs have !!!!! in front of them
    parsed_text = re.sub(emoji_regex,'',parsed_text) #remove emojis from the text
    parsed_text = re.sub('…','',parsed_text) #Remove the special ending character is truncated
    #parsed_text = re.sub('#[\w\-]+', '',parsed_text)
    #parsed_text = parsed_text.code("utf-8", errors='ignore')
    return parsed_text


def preprocess_clean(text_string, remove_hashtags, remove_special_chars):
    # Clean a string down to just text

    parsed_text = preprocess(text_string)
    parsed_text = parsed_text.lower()
    parsed_text = re.sub('\'', '', parsed_text)
    parsed_text = re.sub('|', '', parsed_text)
    parsed_text = re.sub(':', '', parsed_text)
    parsed_text = re.sub(',', '', parsed_text)
    parsed_text = re.sub(';', '.', parsed_text)
    parsed_text = re.sub('&amp', '', parsed_text)

    if remove_hashtags:
        parsed_text = re.sub('#[\w\-]+', '',parsed_text)
    if remove_special_chars:
        #parsed_text = re.sub('(\!|\?)+','.',parsed_text) #find one or more of special char in a row, replace with one '.'
        parsed_text = re.sub('(\!|\?)+','',parsed_text)
    return parsed_text

def strip_hashtags(text):
    d = enchant.Dict('en_UK')
    dus = enchant.Dict('en_US')
    hashtags = re.findall('#[\w\-]+', text)
    for tag in hashtags:
        cleantag = tag[1:]
        print(tag)
        if d.check(cleantag) or dus.check(cleantag):
            text = re.sub(tag,cleantag,text)
            pass
        else:
            hashtagSplit = ""
            for word in splitter.split(cleantag.lower(),'en_US'):
                hashtagSplit = hashtagSplit + word + " "
            print(hashtagSplit)
            text = re.sub(tag,hashtagSplit,text)
    return text