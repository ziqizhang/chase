from ml.vectorizer import fv_davison


# each setting can use a different FeatureVectorizer to create different features. this way we can create a batch of experiments to run
def create_settings():
    sys_out='/home/zqz/Work/chase/output' #where the system will save its required files, such as the trained models
    #data_in='/home/zqz/Work/hate-speech-and-offensive-language/data/labeled_data.csv'
    data_in='/home/zqz/Work/hate-speech-and-offensive-language/data/labeled_data_small.csv'

    settings=[]
    settings.append(['davidson_data+davidson_feature', #just a name to identify this experimental setting
                     'dddf',
                     data_in,
                     fv_davison.FeatureVectorizerDavidson(),
                     sys_out])
    return settings
