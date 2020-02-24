from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import pandas

def wrangle_mtg():
    '''
    Wrangle pandas data frame with relavent columns
    '''

    # read cards.csv into a pandas dataframe
    df = pandas.read_csv('cards.csv')

    # rewite data frame with only relavent columns
    df = df[['colorIdentity','types','convertedManaCost','rarity','flavorText','isPaper']]

    return df

def prepare_mgt(df):
    '''
    Prepare mtg data for analysis
    '''

    # use only cards that exist as phisycal cards
    df = df[df.isPaper==1]
    df = df.drop(columns='isPaper')

    # use only cards with flavor text
    df = df[df.flavorText.notna()]

    # use only cards with a single color identity 
    colors = ['W','U','B','R','G']
    df = df.loc[df.colorIdentity.isin(colors)]

    # merge like card types 

    df['types'] = np.where(df['types'] == 'Artifact,Creature', 'Creature', df['types'])

    df['types'] = np.where(df['types'] == 'Summon', 'Creature', df['types'])

    df['types'] = np.where(df['types'] == 'Land,Creature', 'Land', df['types'])

    df['types'] = np.where(df['types'] == 'Artifact,Land', 'Land', df['types'])

    df['types'] = np.where(df['types'] == 'Tribal,Instant', 'Instant', df['types'])

    df['types'] = np.where(df['types'] == 'Tribal,Sorcery', 'Sorcery', df['types'])

    df['types'] = np.where(df['types'] == 'Tribal,Enchantment', 'Enchantment', df['types'])

    df['types'] = np.where(df['types'] == 'instant', 'Instant', df['types'])

    # remove remaining cards that are not exclusive to one of the seven card types
    types = ['Creature','Instant','Sorcery','Enchantment','Land','Artifact','Planeswalker']
    df = df.loc[df.types.isin(types)]

    # rewrite values in color identity to write the full word
    df['colorIdentity'] = np.where(df['colorIdentity'] == 'G', 'Green', df['colorIdentity'])

    df['colorIdentity'] = np.where(df['colorIdentity'] == 'U', 'Blue', df['colorIdentity'])

    df['colorIdentity'] = np.where(df['colorIdentity'] == 'W', 'White', df['colorIdentity'])

    df['colorIdentity'] = np.where(df['colorIdentity'] == 'B', 'Black', df['colorIdentity'])

    df['colorIdentity'] = np.where(df['colorIdentity'] == 'R', 'Red', df['colorIdentity'])

    return df









analyser = SentimentIntensityAnalyzer()

def sent_score(sentence):
    score = analyser.polarity_scores(sentence)
    print(score)