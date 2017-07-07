'''USE THIS FILE TO TRAIN AND EVALUATE A MODEL'''
import datetime
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import svm
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline

from ml import util
from sklearn.model_selection import GridSearchCV
import os
from time import time
import numpy as np


def create_dimensionality_reducer(option, gridsearch:bool):
    if option==-1:
        #selector=SelectKBest(score_func=chi2, k=200)
        reducer=None
        params = None
    else:
        #selector=PCA(n_components=2, svd_solver='auto')
        reducer=PCA(n_components=2, svd_solver='auto')
        params = {'n_components':[2,3,5], 'svd_solver':['auto']}
    if gridsearch:
        return reducer, params
    else:
        return reducer, {}


def create_classifier(outfolder, model, task, nfold, classifier_gridsearch, featureopt_option,
                      featureopt_gridsearch, cpus):
    classifier = None
    model_file = None
    cl_tuning_params=None
    subfolder=outfolder+"/models"
    try:
        os.stat(subfolder)
    except:
        os.mkdir(subfolder)

    if (model == "rf"):
        print("== Random Forest ...{}".format(datetime.datetime.now()))
        classifier = RandomForestClassifier(n_estimators=20, n_jobs=cpus)
        if classifier_gridsearch:
            cl_tuning_params = {"max_depth": [3, 5, None],
                             "max_features": [1, 3, 5, 7, 10],
                             "min_samples_split": [2, 5, 10],
                             "min_samples_leaf": [1, 3, 10],
                             "bootstrap": [True, False],
                             "criterion": ["gini", "entropy"]}
        else:
            cl_tuning_params={}
        model_file = subfolder+ "/random-forest_classifier-%s.m" % task
    if (model == "svm-l"):
        if classifier_gridsearch:
            cl_tuning_params = {'C': [1e-1, 1e-3, 1e-5, 0.2, 0.5, 1, 1.2, 1.3, 1.5, 1.6, 1.7, 1.8, 2]}
        else:
            cl_tuning_params={}
        print("== SVM, kernel=linear ...{}".format(datetime.datetime.now()))
        classifier = svm.LinearSVC()
        model_file = subfolder+ "/liblinear-svm-linear-%s.m" % task

    if (model == "svm-rbf"):
        if classifier_gridsearch:
            cl_tuning_params = {'gamma': np.logspace(-9, 3, 3), 'probability': [True], 'C': np.logspace(-2, 10, 3)}
        else:
            cl_tuning_params={}
        print("== SVM, kernel=rbf ...{}".format(datetime.datetime.now()))
        classifier = svm.SVC()
        model_file = subfolder+  "/liblinear-svm-rbf-%s.m" % task

    if (model == "sgd"):
        print("== SGD ...{}".format(datetime.datetime.now()))
        #DISABLED because grid search takes too long to complete
        if classifier_gridsearch:
            cl_tuning_params = {"loss": ["log", "modified_huber", 'squared_loss'],
                      "penalty": ['l2', 'l1'],
                      "alpha": [0.0001, 0.001, 0.01, 0.03, 0.05, 0.1],
                      "n_iter": [1000],
                      "learning_rate": ["optimal"]}
        else:
            cl_tuning_params={}
        classifier = SGDClassifier(loss='log', penalty='l2', n_jobs=cpus)
        model_file = subfolder+ "/sgd-classifier-%s.m" % task
    if (model == "lr"):
        print("== Stochastic Logistic Regression ...{}".format(datetime.datetime.now()))
        if classifier_gridsearch:
            cl_tuning_params = {"penalty": ['l2'],
                      "solver": ['liblinear'],
                      "C": list(np.power(10.0, np.arange(-10, 10))),
                      "max_iter": [10000]}
        else:
            cl_tuning_params={}
        classifier = LogisticRegression(random_state=111)
        model_file = subfolder+ "/stochasticLR-%s.m" % task

    dim_reducer=create_dimensionality_reducer(featureopt_option, featureopt_gridsearch)
    if dim_reducer[0] is None:
        pipe = Pipeline([
        ('classify', classifier)])
        all_params=cl_tuning_params
    else:
        pipe = Pipeline([
        ('reduce_dim', dim_reducer[0]),
        ('classify', classifier)])
        all_params=[dim_reducer[1], cl_tuning_params]

    piped_classifier = GridSearchCV(pipe, param_grid=all_params, cv=nfold,
                                  n_jobs=cpus)
    return piped_classifier, model_file


def learn_general(cpus, nfold, task, load_model, model, X_train, y_train, X_test, y_test,
                         identifier, outfolder, classifier_gridsearch=True,
                         featureopt_option=0, featureopt_gridsearch=True):

    c = create_classifier(outfolder,model, task, nfold, classifier_gridsearch,
                      featureopt_option, featureopt_gridsearch, cpus)
    piped_classifier = c[0]
    model_file = c[1]

    nfold_predictions=None

    if load_model:
        print("model is loaded from [%s]" % str(model_file))
        best_estimator = util.load_classifier_model(model_file)
    else:
        piped_classifier.fit(X_train, y_train)
        nfold_predictions=cross_val_predict(piped_classifier.best_estimator_, X_train, y_train, cv=nfold)

        best_estimator = piped_classifier.best_estimator_
        best_param = piped_classifier.best_params_
        print("+ best params for {} model are:{}".format(model, best_param))
        cv_score = piped_classifier.best_score_
        util.save_classifier_model(best_estimator, model_file)

    if(X_test is not None):
        heldout_predictions_final = best_estimator.predict(X_test)
        util.save_scores(nfold_predictions,y_train, heldout_predictions_final, y_test, model, task,
                     identifier, 2,outfolder)
    else:
        util.save_scores(nfold_predictions,y_train, None, y_test, model, task,
                     identifier, 2,outfolder)


def learn_dnn(cpus, nfold, task, load_model, model, input_dim, X_train, y_train, X_test, y_test,
              identifier,outfolder, gridsearch=True,
              featureopt_option=0, featureopt_gridsearch=True):
    print("== Perform ANN ...")  # create model
    subfolder=outfolder+"/models"
    try:
        os.stat(subfolder)
    except:
        os.mkdir(subfolder)
    model = KerasClassifier(build_fn=create_model(input_dim), verbose=0)
    # define the grid search parameters
    batch_size = [10, 20]
    epochs = [50, 100]
    dropout = [0.1, 0.3, 0.5, 0.7]
    param_grid = dict(dropout_rate=dropout, batch_size=batch_size, nb_epoch=epochs)

    dim_reducer=create_dimensionality_reducer(featureopt_option, featureopt_gridsearch)
    if dim_reducer[0] is None:
        pipe = Pipeline([
        ('classify', model)])
        all_params=param_grid
    else:
        pipe = Pipeline([
        ('reduce_dim', dim_reducer[0]),
        ('classify', model)])
        all_params=[dim_reducer[1],param_grid]

    piped_classifier =  GridSearchCV(estimator=pipe, param_grid=all_params, n_jobs=cpus,
                        cv=nfold)

    cv_score_ann = 0
    best_param_ann = []
    ann_model_file = os.path.join(subfolder, "ann-%s.m" % task)
    nfold_predictions=None

    if load_model:
        print("model is loaded from [%s]" % str(ann_model_file))
        best_estimator = util.load_classifier_model(ann_model_file)
    else:
        piped_classifier.fit(X_train, y_train)
        nfold_predictions=cross_val_predict(piped_classifier.best_estimator_, X_train, y_train, cv=nfold)

        cv_score_ann = piped_classifier.best_score_
        best_param_ann = piped_classifier.best_params_
        print("+ best params for {} model are:{}".format(model, best_param_ann))
        best_estimator = piped_classifier.best_estimator_

        # self.save_classifier_model(best_estimator, ann_model_file)

    print("testing on development set ....")
    if(X_test is not None):
        heldout_predictions_final = best_estimator.predict(X_test)
        util.save_scores(nfold_predictions,y_train, heldout_predictions_final, y_test, model, task,
                         identifier,2,outfolder)

    else:
        util.save_scores(nfold_predictions,y_train, None, y_test, model, task, identifier,2,
                         outfolder)

    #util.print_eval_report(best_param_ann, cv_score_ann, dev_data_prediction_ann,
    #                       time_ann_predict_dev,
    #                       time_ann_train, y_test)

def create_model(input_dim,dropout_rate=0.0):
    # create model
    model = Sequential()
    model.add(Dense(80,
                    input_dim=input_dim,
                    init='uniform', activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
