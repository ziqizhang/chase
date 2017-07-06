from ml.vectorizer import fv_davison


# each setting can use a different FeatureVectorizer to create different features. this way we can create a batch of experiments to run
def create_settings(sys_out, data_in):
    #sys_out='../../../output' #where the system will save its required files, such as the trained models
    #data_in='../../../data/labeled_data.csv'
    #data_in='/home/zqz/Work/hate-speech-and-offensive-language/data/labeled_data_small.csv'

    #setting this to true will perform param tuning on the training data,
    #will take significantly longer time. but possibly better results
    USE_GRID_SEARCH=False

    settings=[]
    settings.append(['tdo-fs-scaling-nogs', #task name to identify model files
                     'tdo-fs-scaling-nogs', #identifier to identify scores
                     data_in,
                     fv_davison.FeatureVectorizerDavidson(),#what feature vectorizer to use
                     True, #use feature selection, ONLY VALID FOR TRAINING
                     False, #do grid search for classifier params
                     -1, #use pca for dimensionality reduction
                     False, #grid search on pca
                     sys_out])
    # settings.append(['td_original_noFS', #just a name to identify this experimental setting
    #                   'td_original_noFS',
    #                   data_in,
    #                   fv_davison.FeatureVectorizerDavidson(),
    #                   False,False,
    #                   sys_out])
    return settings
