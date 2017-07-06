import re



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
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, '', parsed_text)
    parsed_text = re.sub(mention_regex, '', parsed_text)
    #parsed_text = re.sub('#[\w\-]+', '',parsed_text)
    #parsed_text = parsed_text.code("utf-8", errors='ignore')
    return parsed_text


#todo: this is based on the above but should allow options to remove hashtags, and special chars
def preprocess_clean(text_string, remove_hashtags, remove_special_chars):
    pass

