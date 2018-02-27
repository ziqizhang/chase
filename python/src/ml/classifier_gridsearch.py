'''USE THIS FILE TO TRAIN AND EVALUATE A MODEL'''
import datetime
import os

import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline

from ml import util
from util import logger as ec

PIPELINE_CLASSIFIER_LABEL = "classify__"
PIPELINE_DIM_REDUCER_LABEL = "dr__"
PIPELINE_FEATURE_SELECTION = "fs__"


def create_dimensionality_reducer(option, gridsearch: bool):
    if option == -1:
        # selector=SelectKBest(score_func=chi2, k=200)
        reducer = None
        params = None
    else:
        # selector=PCA(n_components=2, svd_solver='auto')
        reducer = PCA(n_components=2, svd_solver='arpack')
        params = {
            PIPELINE_DIM_REDUCER_LABEL + 'n_components': [2, 3, 5],
            PIPELINE_DIM_REDUCER_LABEL + 'svd_solver': ['auto', 'arpack']}
    if gridsearch:
        return reducer, params
    else:
        return reducer, {}


def create_feature_selector(option, gridsearch: bool):
    fs = None
    params = None
    if option == 99:
        # selector=SelectKBest(score_func=chi2, k=200)
        return fs, params
    elif option == 0:
        # selector=PCA(n_components=2, svd_solver='auto')
        fs = SelectFromModel(LogisticRegression(class_weight='balanced', penalty="l1", C=0.01))
        params = {}
    elif option == 1:
        fs = SelectKBest(k=1000, score_func=f_classif)
        params = {PIPELINE_FEATURE_SELECTION + 'score_func': [f_classif],
                  PIPELINE_FEATURE_SELECTION + 'k': [100, 250, 500, 1000, 1500, 2000]}
    elif option == 2:
        # fs=RandomizedLogisticRegression(n_jobs=4, random_state=42)
        # params = {PIPELINE_FEATURE_SELECTION+'sample_fraction':[0.3,0.5],
        #           PIPELINE_FEATURE_SELECTION+'selection_threshold':[0.25,0.5]}
        logreg = LogisticRegression()
        # Use RFECV to pick best features, using Stratified Kfold
        fs = RFECV(estimator=logreg, step=250, cv=5, scoring='accuracy')
        params = {}

    if gridsearch:
        return fs, params
    else:
        return fs, {}


def create_classifier(outfolder, model, task, nfold, classifier_gridsearch, dr_option,
                      dr_gridsearch, fs_option, fs_gridsearch, cpus):
    classifier = None
    model_file = None
    cl_tuning_params = None
    subfolder = outfolder + "/models"
    try:
        os.stat(subfolder)
    except:
        os.mkdir(subfolder)

    if (model == "rf"):
        ec.logger.info("== Random Forest ...{}".format(datetime.datetime.now()))
        classifier = RandomForestClassifier(n_estimators=20, n_jobs=cpus)
        if classifier_gridsearch:
            cl_tuning_params = {PIPELINE_CLASSIFIER_LABEL + "max_depth": [3, 5, None],
                                PIPELINE_CLASSIFIER_LABEL + "max_features": [1, 3, 5, 7, 10],
                                PIPELINE_CLASSIFIER_LABEL + "min_samples_split": [2, 5, 10],
                                PIPELINE_CLASSIFIER_LABEL + "min_samples_leaf": [1, 3, 10],
                                PIPELINE_CLASSIFIER_LABEL + "bootstrap": [True, False],
                                PIPELINE_CLASSIFIER_LABEL + "criterion": ["gini", "entropy"]}
        else:
            cl_tuning_params = {}
        model_file = subfolder + "/random-forest_classifier-%s.m" % task
    if (model == "svml"):
        if classifier_gridsearch:
            cl_tuning_params = {
                PIPELINE_CLASSIFIER_LABEL + 'C': [1e-1, 1e-3, 1e-5, 0.2, 0.5, 1, 1.2, 1.3, 1.5, 1.6, 1.7, 1.8, 2]}
        else:
            cl_tuning_params = {}
        ec.logger.info("== SVM, kernel=linear ...{}".format(datetime.datetime.now()))
        classifier = svm.LinearSVC(class_weight='balanced', C=0.01, penalty='l2', loss='squared_hinge',
                                   multi_class='ovr')
        model_file = subfolder + "/svml-%s.m" % task

    if (model == "svmrbf"):
        if classifier_gridsearch:
            cl_tuning_params = {PIPELINE_CLASSIFIER_LABEL + 'gamma': np.logspace(-9, 3, 3),
                                PIPELINE_CLASSIFIER_LABEL + 'probability': [True],
                                PIPELINE_CLASSIFIER_LABEL + 'C': np.logspace(-2, 10, 3)}
        else:
            cl_tuning_params = {}
        ec.logger.info("== SVM, kernel=rbf ...{}".format(datetime.datetime.now()))
        classifier = svm.SVC()
        model_file = subfolder + "/svmrbf-%s.m" % task

    if (model == "sgd"):
        ec.logger.info("== SGD ...{}".format(datetime.datetime.now()))
        # DISABLED because grid search takes too long to complete
        if classifier_gridsearch:
            cl_tuning_params = {PIPELINE_CLASSIFIER_LABEL + "loss": ["log", "modified_huber", 'squared_loss'],
                                PIPELINE_CLASSIFIER_LABEL + "penalty": ['l2', 'l1'],
                                PIPELINE_CLASSIFIER_LABEL + "alpha": [0.0001, 0.001, 0.01, 0.03, 0.05, 0.1],
                                PIPELINE_CLASSIFIER_LABEL + "n_iter": [1000],
                                PIPELINE_CLASSIFIER_LABEL + "learning_rate": ["optimal"]}
        else:
            cl_tuning_params = {}
        classifier = SGDClassifier(loss='log', penalty='l2', n_jobs=cpus)
        model_file = subfolder + "/sgd-classifier-%s.m" % task
    if (model == "lr"):
        ec.logger.info("== Stochastic Logistic Regression ...{}".format(datetime.datetime.now()))
        if classifier_gridsearch:
            cl_tuning_params = {
                PIPELINE_CLASSIFIER_LABEL + "penalty": ['l2'],
                PIPELINE_CLASSIFIER_LABEL + "solver": ['liblinear'],
                PIPELINE_CLASSIFIER_LABEL + "C": list(np.power(10.0, np.arange(-10, 10))),
                PIPELINE_CLASSIFIER_LABEL + "max_iter": [10000]}
        else:
            cl_tuning_params = {}
        classifier = LogisticRegression(random_state=111)
        model_file = subfolder + "/stochasticLR-%s.m" % task

    dim_reducer = create_dimensionality_reducer(dr_option, dr_gridsearch)
    feature_selector = create_feature_selector(fs_option, fs_gridsearch)
    pipe = []
    params = []
    if feature_selector[0] is not None:
        pipe.append(('fs', feature_selector[0]))
        params.append(feature_selector[1])
    if dim_reducer[0] is not None:
        pipe.append(('dr', dim_reducer[0]))
        params.append(dim_reducer[1])
    pipe.append(('classify', classifier))
    params.append(cl_tuning_params)

    pipeline = Pipeline(pipe)
    piped_classifier = GridSearchCV(pipeline, param_grid=params, cv=nfold,
                                    n_jobs=cpus)
    return piped_classifier, model_file


def learn_general(cpus, nfold, task, load_model, model,
                  feature_vocbs: dict, X_train, y_train, X_test, y_test,
                  index_train, index_test,
                  outfolder, classifier_gridsearch=True,
                  dr_option=0, dr_gridsearch=True, fs_option=0, fs_gridsearch=True,
                  instance_data_source_tags=None, accepted_ds_tags: list = None
                  ):
    c = create_classifier(outfolder, model, task, nfold, classifier_gridsearch,
                          dr_option, dr_gridsearch, fs_option, fs_gridsearch, cpus)
    piped_classifier = c[0]
    model_file = c[1]

    # print("### test sfm...")
    # fs=SelectFromModel(LogisticRegression(class_weight='balanced', penalty="l1", C=0.01))
    # X_=fs.fit_transform(X_train, y_train)
    # print(np.count_nonzero(X_))
    # print(X_.shape)
    # print("### end \n\n\n")
    #
    # print("### test kb...")
    # fs=SelectKBest(k=1000, score_func=f_classif)
    # X_=fs.fit_transform(X_train, y_train)
    # print(np.count_nonzero(X_))
    # print(X_.shape)
    # print("### end \n\n\n")

    nfold_predictions = None

    if load_model:
        ec.logger.info("model is loaded from [%s]" % str(model_file))
        best_estimator = util.load_classifier_model(model_file)
    else:
        piped_classifier.fit(X_train, y_train)
        nfold_predictions = cross_val_predict(piped_classifier.best_estimator_, X_train, y_train, cv=nfold)

        best_estimator = piped_classifier.best_estimator_
        best_param = piped_classifier.best_params_
        ec.logger.info("+ best params for {} model are:{}".format(model, best_param))
        cv_score = piped_classifier.best_score_
        util.save_classifier_model(best_estimator, model_file)

        # selected features for inspection
        if 'fs' in best_estimator.named_steps.keys():
            finalFeatureIndices = best_estimator.named_steps["fs"].get_support(indices=True)
            util.save_selected_features(finalFeatureIndices, feature_vocbs,
                                        model_file + ".features.csv")

    if (X_test is not None):
        heldout_predictions_final = best_estimator.predict(X_test)
        util.save_scores(nfold_predictions, y_train,
                         heldout_predictions_final,
                         y_test,index_train, index_test,
                         model, task,
                         2, outfolder,
                         instance_data_source_tags, accepted_ds_tags)

    else:
        util.save_scores(nfold_predictions, y_train, None, y_test, model, task,
                         2, outfolder,
                         instance_data_source_tags, accepted_ds_tags)
