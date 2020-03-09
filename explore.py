import pandas as pd
import numpy as np
import math

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression

import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


def get_value_counts(df):
    '''
    Print value Count for features in data frame
    '''

    for column in df.columns:
    
        if column not in ('sentiment','flavor','intensity'):
            print(f'{column} value counts')
            print(df[f'{column}'].value_counts())
            print('')

def sent_percent(df):
    '''
    Print percent of positive negative and zore sentiment scores 
    '''

    print("Positive and Negative Sentiment Scores")
    print('')

    positive = df[df.sentiment>0].sentiment.count()
    negative = df[df.sentiment<0].sentiment.count()
    zero = df[df.sentiment==0].sentiment.count()
    
    total = df.sentiment.count()
          
    print(f'Positive: {round(positive/total,2)}%  Negative: {round(negative/total,2)}%  Zero: {round(zero/total,2)}%')


def sent_dist(df):
    '''
    Print sentiment distribution for full data set with and without zero scores included
    '''
    
    bins =[-1,-.9,-.8,-.7,-.6,-.5,-.4,-.3,-.2,-.1,0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]

    sentiment = df['sentiment']

    plt.hist(sentiment,bins=bins,edgecolor='black')


    plt.title('Fequency of Sentiment Scores')
    plt.xlabel('Sentiment Scores')
    plt.ylabel('Number of Occurances')

    plt.tight_layout()

    plt.show()

    sentiment = df.sentiment[df.sentiment!=0]

    plt.hist(sentiment,bins=bins,edgecolor='black')


    plt.title('Fequency of Sentiment Scores With Zeros Removed')
    plt.xlabel('Sentiment Scores')
    plt.ylabel('Number of Occurances')

    plt.tight_layout()

    plt.show()

def int_dist(df):
    '''
    Print intensity distribution for full data set with and without zero scores included
    '''

    bins =[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]

    intensity = df['intensity']

    plt.hist(intensity,bins=bins,edgecolor='black')


    plt.title('Fequency of Intensity Scores')
    plt.xlabel('Sentiment Scores')
    plt.ylabel('Number of Occurances')

    plt.tight_layout()
    plt.show()

    intensity = df.intensity[df.intensity!=0]

    plt.hist(intensity,bins=bins,edgecolor='black')


    plt.title('Fequency of Intensity Scores With Zeros Removed')
    plt.xlabel('Sentiment Scores')
    plt.ylabel('Number of Occurances')

    plt.tight_layout()
    plt.show()

def get_scores_color(df):
        '''
        Print mean, median, and mode sentiment scores by color
        '''

        colors = ['White','Blue','Black','Red','Green']

        print('Mean Sentiment by Color')

        for color in colors:

            number = df[df.color==f'{color}'].sentiment.mean()
      
            print(f'{color}: {round(number,2)}')

        print('')
        print('Median Sentiment by Color')

        for color in colors:

            number = df[df.color==f'{color}'].sentiment.median()
      
            print(f'{color}: {round(number,2)}')

        print('')

        print('Mode Sentiment by Color')

        for color in colors:

            number = df[df.color==f'{color}'].sentiment.median()
      
            print(f'{color}: {round(number,2)}')


def sent_percent_color(df):

    colors = ['White','Blue','Black','Red','Green']

    print("Positive and Negative Sentiment Scores by Color")

    for color in colors:

        positive = df[df.color==f'{color}'][df.sentiment>0].sentiment.count()
        negative = df[df.color==f'{color}'][df.sentiment<0].sentiment.count()
        zero = df[df.color==f'{color}'][df.sentiment==0].sentiment.count()
    
        total = df[df.color==f'{color}'].sentiment.count()
      
        print('')
        print(f'{color}: Positive: {round(positive/total,2)}%  Negative: {round(negative/total,2)}%  Zero: {round(zero/total,2)}%')


def sent_dist_color(df):

    colors = ['White','Blue','Black','Red','Green']

    bins =[-1,-.9,-.8,-.7,-.6,-.5,-.4,-.3,-.2,-.1,0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]

    print("Distribution of Sentiment Scores by Color")

    for color in colors:

        sentiment = df.sentiment[df.sentiment!=0][df.color==f'{color}']

        plt.hist(sentiment,bins=bins,edgecolor='black',color=f'{color}')

        plt.title(f'Fequency of Sentiment Scores in {color} With Zero Scores Removed')
        plt.xlabel('Sentiment Scores')
        plt.ylabel('Number of Occurances')

        plt.tight_layout()

        plt.show()

def get_scores_type(df):
    '''
    Print mean, median, and mode sentiment scores by type
    '''

    print('Mean Sentiment by Type')

    types = ['Creature','Instant','Sorcery','Enchantment','Land','Artifact','Planeswalker']

    for item in types:

        number = df[df.type==f'{item}'].sentiment.mean()
      
        print(f'{item}: {round(number,2)}')

    
    print('')
    print('Median Sentiment by Type')

    for item in types:

        number = df[df.type==f'{item}'].sentiment.median()
      
        print(f'{item}: {round(number,2)}')

    print('')
    print('Mode Sentiment by Type')

    for item in types:

        number = df[df.type==f'{item}'].sentiment.mode()
      
        print(f'{item}: {round(number,2)}')

def sent_percent_type(df):

    types = ['Creature','Instant','Sorcery','Enchantment','Land','Artifact','Planeswalker']

    print("Distribution of Sentiment Scores by Type")

    for item in types:

        positive = df[df.type==f'{item}'][df.sentiment>0].sentiment.count()
        negative = df[df.type==f'{item}'][df.sentiment<0].sentiment.count()
        zero = df[df.type==f'{item}'][df.sentiment==0].sentiment.count()
    
        total = df[df.type==f'{item}'].sentiment.count()
      
        print('')
        print(f'{item}: Positive: {round(positive/total,2)}%  Negative: {round(negative/total,2)}%  Zero: {round(zero/total,2)}%')


def sent_dist_type(df):

    types = ['Creature','Instant','Sorcery','Enchantment','Land','Artifact','Planeswalker']
    bins =[-1,-.9,-.8,-.7,-.6,-.5,-.4,-.3,-.2,-.1,0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]

    for item in types:

        sentiment = df.sentiment[df.sentiment!=0][df.type==f'{item}']

        plt.hist(sentiment,bins=bins,edgecolor='black',color='mistyrose')

        plt.title(f'Fequency of Sentiment Scores in {item} With Zero Scores Removed')
        plt.xlabel('Sentiment Scores')
        plt.ylabel('Number of Occurances')

        plt.tight_layout()

        plt.show()