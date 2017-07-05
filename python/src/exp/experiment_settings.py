from ml.vectorizer import fv_davison


# each setting can use a different FeatureVectorizer to create different features. this way we can create a batch of experiments to run
def create_settings():
    sys_out='/home/zqz/Work/chase/output' #where the system will save its required files, such as the trained models
    data_in='/home/zqz/Work/hate-speech-and-offensive-language/data/labeled_data.csv'
    #data_in='/home/zqz/Work/hate-speech-and-offensive-language/data/labeled_data_small.csv'

    #setting this to true will perform param tuning on the training data,
    #will take significantly longer time. but possibly better results
    USE_GRID_SEARCH=False

    settings=[]
    settings.append(['td_original', #just a name to identify this experimental setting
                     'td_original',
                     data_in,
                     fv_davison.FeatureVectorizerDavidson(),
                     True,USE_GRID_SEARCH,
                     sys_out])
    settings.append(['td_original_noFS', #just a name to identify this experimental setting
                      'td_original_noFS',
                      data_in,
                      fv_davison.FeatureVectorizerDavidson(),
                      False,USE_GRID_SEARCH,
                      sys_out])
    return settings
