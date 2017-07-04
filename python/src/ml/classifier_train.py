'''USE THIS FILE TO TRAIN AND EVALUATE A MODEL'''
import datetime
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import svm

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict

from ml import util
from sklearn.model_selection import GridSearchCV
import os
from time import time
import numpy as np


def learn_discriminative(cpus, nfold, task, load_model, model, X_train, y_train, X_test, y_test,
                         identifier, outfolder):
    classifier = None
    model_file = None
    subfolder=outfolder+"/models"
    try:
        os.stat(subfolder)
    except:
        os.mkdir(subfolder)

    if (model == "rf"):
        print("== Random Forest ...{}".format(datetime.datetime.now()))
        classifier = RandomForestClassifier(n_estimators=20, n_jobs=cpus)
        rfc_tuning_params = {"max_depth": [3, 5, None],
                             "max_features": [1, 3, 5, 7, 10],
                             "min_samples_split": [2, 5, 10],
                             "min_samples_leaf": [1, 3, 10],
                             "bootstrap": [True, False],
                             "criterion": ["gini", "entropy"]}
        classifier = GridSearchCV(classifier, param_grid=rfc_tuning_params, cv=nfold,
                                  n_jobs=cpus)
        model_file = os.path.join(subfolder, "random-forest_classifier-%s.m" % task)
    if (model == "svm-l"):
        tuned_parameters = [{'gamma': np.logspace(-9, 3, 3), 'probability': [True], 'C': np.logspace(-2, 10, 3)},
                            {'C': [1e-1, 1e-3, 1e-5, 0.2, 0.5, 1, 1.2, 1.3, 1.5, 1.6, 1.7, 1.8, 2]}]
        print("== SVM, kernel=linear ...{}".format(datetime.datetime.now()))
        classifier = svm.LinearSVC()
        classifier = GridSearchCV(classifier, tuned_parameters[1], cv=nfold, n_jobs=cpus)
        model_file = os.path.join(subfolder, "liblinear-svm-linear-%s.m" % task)

    if (model == "svm-rbf"):
        tuned_parameters = [{'gamma': np.logspace(-9, 3, 3), 'probability': [True], 'C': np.logspace(-2, 10, 3)},
                             {'C': [1e-1, 1e-3, 1e-5, 0.2, 0.5, 1, 1.2, 1.3, 1.5, 1.6, 1.7, 1.8, 2]}]
        print("== SVM, kernel=rbf ...{}".format(datetime.datetime.now()))
        classifier = svm.SVC()
        classifier = GridSearchCV(classifier, param_grid=tuned_parameters[0], cv=nfold, n_jobs=cpus)
        model_file = os.path.join(subfolder, "liblinear-svm-rbf-%s.m" % task)

    best_param = []
    cv_score = 0
    best_estimator = None
    nfold_predictions=None

    t0 = time()
    if load_model:
        print("model is loaded from [%s]" % str(model_file))
        best_estimator = util.load_classifier_model(model_file)
    else:
        classifier.fit(X_train, y_train)
        nfold_predictions=cross_val_predict(classifier.best_estimator_, X_train, y_train, cv=nfold)

        best_estimator = classifier.best_estimator_
        best_param = classifier.best_params_
        cv_score = classifier.best_score_
        util.save_classifier_model(best_estimator, model_file)

    if(X_test is not None):
        heldout_predictions_final = best_estimator.predict(X_test)
        util.save_scores(nfold_predictions,y_train, heldout_predictions_final, y_test, model, task,
                     identifier, 2,outfolder)
    else:
        util.save_scores(nfold_predictions,y_train, None, y_test, model, task,
                     identifier, 2,outfolder)



def learn_generative(cpus, nfold, task, load_model, model, X_train, y_train, X_test, y_test,
                     identifier, outfolder):
    classifier = None
    model_file = None
    subfolder=outfolder+"/models"
    try:
        os.stat(subfolder)
    except:
        os.mkdir(subfolder)

    if (model == "sgd"):
        print("== SGD ...{}".format(datetime.datetime.now()))
        #DISABLED because grid search takes too long to complete
        sgd_params = {"loss": ["log", "modified_huber", 'squared_loss'],
                      "penalty": ['l2', 'l1'],
                      "alpha": [0.0001, 0.001, 0.01, 0.03, 0.05, 0.1],
                      "n_iter": [1000],
                      "learning_rate": ["optimal"]}
        classifier = SGDClassifier(loss='log', penalty='l2', n_jobs=cpus)

        classifier = GridSearchCV(classifier, param_grid=sgd_params, cv=nfold,
                                  n_jobs=cpus)
        model_file = os.path.join(subfolder, "sgd-classifier-%s.m" % task)
    if (model == "lr"):
        print("== Stochastic Logistic Regression ...{}".format(datetime.datetime.now()))
        slr_params = {"penalty": ['l2'],
                      "solver": ['liblinear'],
                      "C": list(np.power(10.0, np.arange(-10, 10))),
                      "max_iter": [10000]}
        classifier = LogisticRegression(random_state=111)
        classifier = GridSearchCV(classifier, param_grid=slr_params, cv=nfold,
                                  n_jobs=cpus)
        model_file = os.path.join(subfolder, "stochasticLR-%s.m" % task)

    best_param = []
    cv_score = 0
    best_estimator = None
    nfold_predictions=None

    if load_model:
        print("model is loaded from [%s]" % str(model_file))
        best_estimator = util.load_classifier_model(model_file)
    else:
        classifier.fit(X_train, y_train)
        nfold_predictions=cross_val_predict(classifier.best_estimator_, X_train, y_train, cv=nfold)

        best_estimator = classifier.best_estimator_
        best_param = classifier.best_params_
        cv_score = classifier.best_score_
        util.save_classifier_model(best_estimator, model_file)
    classes = classifier.best_estimator_.classes_

    if(X_test is not None):
        heldout_predictions = best_estimator.predict_proba(X_test)
        heldout_predictions_final = [classes[util.index_max(list(probs))] for probs in heldout_predictions]
        util.save_scores(nfold_predictions,y_train, heldout_predictions_final, y_test, model, task,
                     identifier, 2,outfolder)
    else:
        util.save_scores(nfold_predictions,y_train, None, y_test, model, task, identifier, 2,outfolder)


def learn_dnn(cpus, nfold, task, load_model, model, input_dim, X_train, y_train, X_test, y_test,
              identifier,outfolder):
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
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=cpus,
                        cv=nfold)

    t0 = time()
    cv_score_ann = 0
    best_param_ann = []
    ann_model_file = os.path.join(subfolder, "ann-%s.m" % task)
    nfold_predictions=None

    if load_model:
        print("model is loaded from [%s]" % str(ann_model_file))
        best_estimator = util.load_classifier_model(ann_model_file)
    else:
        grid.fit(X_train, y_train)
        nfold_predictions=cross_val_predict(grid.best_estimator_, X_train, y_train, cv=nfold)

        cv_score_ann = grid.best_score_
        best_param_ann = grid.best_params_
        best_estimator = grid.best_estimator_

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
