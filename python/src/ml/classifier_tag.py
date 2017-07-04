'''USE THIS FILE TO APPLY PRE-TRAINED MODEL TO TAG DATA'''

from ml import util
import os

def tag(cpus, model, task, test_data):
    print("start testing stage :: testing data size:", len(test_data))
    print("test with CPU cores: [%s]" % cpus)

    ######################### SGDClassifier #######################
    model_file=None
    if model=="sgd":
        # SGD doesn't work so well with only a few samples, but is (much more) performant with larger data
        # At n_iter=1000, SGD should converge on most datasets
        print("Using SGD ...")
        model_file = os.path.join(os.path.dirname(__file__), "sgd-classifier-%s.m" % task)

    ######################### Stochastic Logistic Regression#######################
    if model=="lr":
        print("Using Stochastic Logistic Regression ...")
        model_file = os.path.join(os.path.dirname(__file__), "stochasticLR-%s.m" % task)

    ######################### Random Forest Classifier #######################
    if model=="rf":
        print("Using Random Forest ...")
        model_file = os.path.join(os.path.dirname(__file__), "random-forest_classifier-%s.m" % task)

    ###################  liblinear SVM ##############################
    if model=="svm-l":
        print("Using SVM, kernel=linear ...")
        model_file = os.path.join(os.path.dirname(__file__), "liblinear-svm-linear-%s.m" % task)

    ##################### RBF svm #####################
    if model=="svm-rbf":
        print("Using SVM, kernel=rbf ....")
        model_file = os.path.join(os.path.dirname(__file__), "liblinear-svm-rbf-%s.m" % task)

    best_estimator = util.load_classifier_model(model_file)
    prediction_dev = best_estimator.predict_proba(test_data)
    util.saveOutput(prediction_dev, model)
