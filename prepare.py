from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()

def sent_score(sentence):
    score = analyser.polarity_scores(sentence)
    print(score)