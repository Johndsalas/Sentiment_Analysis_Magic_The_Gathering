
import pandas as pd
import numpy as np
import math
from math import sqrt

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, explained_variance_score

import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

def get_baseline(df):
    '''
    Get baseline error stats and explained variance score using mean of sentiment 
    '''

    # add mean sentiment to data frame
    df['yhat'] = df.sentiment.mean()

    # define y and yhat
    yhat = df[['yhat']]

    y = df[['sentiment']]

    # create and fit regression model
    lm1 = LinearRegression()
    lm1.fit(yhat, y)

    # conver yhat to numpy array
    y_pred_baseline = np.array(yhat)

    # generatr error stats
    MSE = mean_squared_error(y, y_pred_baseline)
    SSE = MSE*len(y)
    RMSE = sqrt(MSE)

    # generate explained variance score
    evs = explained_variance_score(y, y_pred_baseline)

    # print results
    print("Baseline Scores")
    print('')
    print(f"Mean Squared Error: {MSE}")
    print('')
    print(f"Sum of Squared Errors: {SSE}")
    print('')
    print(f"Root Mean Square Errors: {RMSE}")
    print('')
    print(f"Explained Variance Score: {evs}")


def encode_color(X):
    '''
    encode color feature 
    '''

    encoded_values = ['Black','Blue','Green','Red','White']

    # Integer Encoding
    int_encoder = LabelEncoder()
    X.encoded = int_encoder.fit_transform(X.color)

    # create 2D np arrays of the encoded variable (in train and test)
    X_array = np.array(X.encoded).reshape(len(X.encoded),1)

    # One Hot Encoding
    ohe = OneHotEncoder(sparse=False, categories='auto')
    X_ohe = ohe.fit_transform(X_array)

    # Turn the array of new values into a data frame with columns names being the values
    # and index matching that of X
    X_encoded = pd.DataFrame(data=X_ohe, columns=encoded_values, index=X.index)

    # then merge the new dataframe with the existing train/test dataframe
    X = X.join(X_encoded)

    # drop color from the data frame
    X.drop(columns='color',inplace=True)

    return X


def get_regression(df):
    '''
    Get baseline error stats and explained variance score using mean of sentiment  
    '''

    # define X and y
    y = y = df[['sentiment']]
    X = df[['color']]

    # encode X
    X = encode_color(X)

    # create and fit regression object
    lm1 = LinearRegression()
    lm1.fit(X, y)

    # make predictions
    y_pred_color = lm1.predict(X)

    # generate error stats
    MSE = mean_squared_error(y, y_pred_color)
    SSE = MSE*len(y)
    RMSE = sqrt(MSE)

    # generate explaned variance
    evs = explained_variance_score(y, y_pred_color)

    # print results
    print('Model Scores')
    print('')
    print(f"Mean Squared Error: {MSE}")
    print('')
    print(f"Sum of Squared Errors: {SSE}")
    print('')
    print(f"Root Mean Square Errors: {RMSE}")
    print('')
    print(f"Explained Variance Score: {round(evs,2)}")