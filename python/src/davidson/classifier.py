"""
This file contains code to

    (a) Load the pre-trained classifier and
    associated files.

    (b) Transform new input data into the
    correct format for the classifier.

    (c) Run the classifier on the transformed
    data and return results.
"""

import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

from ml import text_preprocess as tp
from ml.vectorizer import fv_davison as fv

OUT_FOLDER="../../../output"

def predictions(X, model):
    """
    This function calls the predict function on
    the trained model to generated a predicted y
    value for each observation.
    """
    y_preds = model.predict(X)
    return y_preds


if __name__ == '__main__':
    print ("Loading data to classify...")

    #Tweets obtained here: https://github.com/sashaperigo/Trump-Tweets
    df = pd.read_csv('../../../data/labeled_data.csv', sep=',')
    tweets = df.tweet
    tweets = [x for x in tweets if type(x) == str]
    print (len(tweets), " tweets to classify")

    #Construct tfidf matrix and get relevant scores
    tweets_cleaned = [tp.preprocess(x) for x in tweets]

    fvobj = fv.FeatureVectorizerDavidson()
    M = fvobj.transform_inputs(tweets, tweets_cleaned, OUT_FOLDER,'none')[0]


    X = pd.DataFrame(M)
    y = df['class'].astype(int)
    select = SelectFromModel(LogisticRegression(class_weight='balanced',penalty="l1",C=0.01))
    X_ = select.fit_transform(X,y)

    model = LinearSVC(class_weight='balanced',C=0.01, penalty='l2', loss='squared_hinge',multi_class='ovr').fit(X_, y)
    y_preds = model.predict(X_)
    report = classification_report( y, y_preds )
    print(report)


    # print ("Loading trained classifier... ")
    # model = joblib.load('final_model.pkl')
    #
    # print ("Loading other information...")
    # tf_vectorizer = joblib.load('final_tfidf.pkl')
    # idf_vector = joblib.load('final_idf.pkl')
    # pos_vectorizer = joblib.load('final_pos.pkl')
    # #Load ngram dict
    # #Load pos dictionary
    # #Load function to transform data
    #
    # print ("Transforming inputs...{}".format(datetime.datetime.now()))
    # X = transform_inputs(tweets, tf_vectorizer, idf_vector, pos_vectorizer)
    #
    # print ("Running classification model...{}".format(datetime.datetime.now()))
    # y = predictions(X, model)
    #
    # print ("Printing predicted values: {}".format(datetime.datetime.now()))
    # for i,t in enumerate(tweets):
    #     print (t)
    #     print (class_to_name(y[i]))
