from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd
import re
import pathlib

def get_mtg_data():

    file = pathlib.Path("mtgprep.csv")

    if file.exists ():

        df = pd.read_csv('mtgprep.csv')

    else:
        
        df = prepare_mtg(wrangle_mtg())

        df.to_csv('mtgprep.csv', index=False)

    return df




def wrangle_mtg():
    '''
    Wrangle pandas data frame with relavent columns
    '''

    # read cards.csv into a pandas dataframe
    df = pd.read_csv('cards.csv')

    # rewite data frame with only relavent columns
    df = df[['colorIdentity','types','convertedManaCost','rarity','flavorText','isPaper']]

    return df

def prepare_mtg(df):
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

    # clean up flavorText and groupby flavorText to reduce duplicates
    df['flavorText'] = df.flavorText.apply(remove)

    df['flavorText'] = df.flavorText.apply(erase_end)

    df['flavorText'] = df.flavorText.apply(remove_space)

    df = df.groupby('flavorText').agg('max').reset_index()

    # reorder columns 
    df = df[['colorIdentity','types','convertedManaCost','rarity','flavorText']]

    # remove rows with flavor text no in English
    df = df.drop([12450,12451,12453,12454,12455,12456,12457,12458,12459,12460,12461,12462])

    # remove seen duplicates
    df = df.drop([2,7968,6562])

    # rename columns 
    df=df.rename(columns={'colorIdentity':'color','convertedManaCost':'cost','flavorText':'flavor'})

    # add sentament and intensity columns
    df['sentiment'] = df.flavorText.apply(sent_score)

    df['intensity'] = df.sentiment.abs()

    

    return df

def remove(value):
    '''
    removes / from text
    '''

    return re.sub(r"[/]",'', value)

def remove_space(value):
    '''
    remove whitespace from around text
    '''
    
    return value.strip()

def erase_end(value):
    '''
    remove quote attribution

    '''
 
    count = 0
    index = 0
    quote = []

    # itterate through characters in string
    for letter in value:
    
        count += 1

        # if character is a " append it to quote for a count
        if letter == '\"':
            quote.append(letter)

            # if quote has a length of 2 set index number and stop counting
            if len(quote) == 2: 
                index = count
                break

 # if quotes exist return string up to the end of the quote
    if len(quote) >= 2: 
        
        return value[:index]
    
    # if quotes dont exist return value unchanged
    else:

        return value

analyser = SentimentIntensityAnalyzer()

def sent_score(sentence):
    '''
    get compound score using vader sentament analysis 
    '''
    
    # define analyser
    analyser = SentimentIntensityAnalyzer()
    
    # get sentament score
    score = analyser.polarity_scores(sentence)

    # return only compound score
    return score['compound']



df = get_mtg_data()

df.head()