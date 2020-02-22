from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas

def wrangle_mtg():
    '''
    Wrangle pandas data frame with relavent columns
    '''

    # read cards.csv into a pandas dataframe
    df = pandas.read_csv('cards.csv')

    # rewite data frame with only relavent colors
    df = df[['name','colorIdentity','colors','convertedManaCost','flavorText','isPaper','rarity','subtypes','supertypes','types']]

    # use only cards that exist as phisycal cards
    df = df[df.isPaper==1]
    df = df.drop(columns='isPaper')

    return df

analyser = SentimentIntensityAnalyzer()

def sent_score(sentence):
    score = analyser.polarity_scores(sentence)
    print(score)