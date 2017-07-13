from ml.vectorizer import fv_davison


def create_settings(sys_out, data_train, data_test):
    #sys_out='../../../output' #where the system will save its required files, such as the trained models
    #data_in='../../../data/labeled_data.csv'
    #data_in='/home/zqz/Work/hate-speech-and-offensive-language/data/labeled_data_small.csv'

    #setting this to true will perform param tuning on the training data,
    #will take significantly longer time. but possibly better results

    settings=[]
    settings.append(['tdo-original-no_gs', #task name to identify model files
                     'tdo-original-no_gs', #identifier to identify scores
                     data_train,
                     data_test,
                     fv_davison.FeatureVectorizerDavidson(),#what feature vectorizer to use
                     0, #fs option
                     sys_out])


    # settings.append(['tdo-scaling-gs', #task name to identify model files
    #                  'tdo-scaling-gs', #identifier to identify scores
    #                  data_in,
    #                  fv_davison.FeatureVectorizerDavidson(),#what feature vectorizer to use
    #                  False, #use feature selection, ONLY VALID FOR TRAINING
    #                  True, #do grid search for classifier params
    #                  -1, #use pca for dimensionality reduction
    #                  False, #grid search on pca
    #                  sys_out])
    return settings
