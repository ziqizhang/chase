
from ml.vectorizer import feature_vectorizer as fv

# each setting can use a different FeatureVectorizer to create different features. this way we can create a batch of experiments to run
def create_settings(sys_out, data_in, label, scores_per_ds, fvect: fv.FeatureVectorizer,
                    fs_options):
    #sys_out='../../../output' #where the system will save its required files, such as the trained models
    #data_in='../../../data/labeled_data.csv'
    #data_in='/home/zqz/Work/hate-speech-and-offensive-language/data/labeled_data_small.csv'

    #setting this to true will perform param tuning on the training data,
    #will take significantly longer time. but possibly better results
    USE_GRID_SEARCH=False

    settings=[]
    if "none" in fs_options:
        settings.append([label, #task name to identify model files
                         label, #identifier to identify scores
                         data_in,
                         fvect,#what feature vectorizer to use
                         False, #do grid search for classifier params
                         -1, #use pca for dimensionality reduction
                         False, #grid search on pca
                         99, #99 means no fs
                         False, #grid search for fs
                         sys_out,
                         scores_per_ds]) #if mixed dataset, set to true to output scores for tweets from each dataset
    if "kb" in fs_options:
        settings.append(['{}-kb'.format(label), #task name to identify model files
                         '{}-kb'.format(label), #identifier to identify scores
                         data_in,
                         fvect,#what feature vectorizer to use
                         False, #do grid search for classifier params
                         -1, #use pca for dimensionality reduction
                         False, #grid search on pca
                         1, #99 means no fs
                         True,
                         sys_out,
                         scores_per_ds])
    if "sfm" in fs_options:
        settings.append(['{}-sfm'.format(label), #task name to identify model files
                         '{}-sfm'.format(label), #identifier to identify scores
                         data_in,
                         fvect,#what feature vectorizer to use
                         False, #do grid search for classifier params
                         -1, #use pca for dimensionality reduction
                         False, #grid search on pca
                         0, #99 means no fs
                         False,
                         sys_out,
                         scores_per_ds])
    # settings.append(['{}-rfe'.format(label), #task name to identify model files
    #                  '{}-rfe'.format(label), #identifier to identify scores
    #                  data_in,
    #                  fvect,#what feature vectorizer to use
    #                  False, #do grid search for classifier params
    #                  -1, #use pca for dimensionality reduction
    #                  False, #grid search on pca
    #                  2, #99 means no fs
    #                  True,
    #                  sys_out,
    #                  scores_per_ds])
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
