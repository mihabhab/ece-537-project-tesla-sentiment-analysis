import os
import pandas as pd
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler

# *** FUNCTION DEFINITION ***
def assign_sentiment(polarity, sentiment_list, num_segments):
    # Calculate the width of each segment
    segment_width = 1 / num_segments
    
    # Determine the sentiment based on polarity
    for i in range(num_segments):
        if i == num_segments - 1:
            if polarity >= i * segment_width and polarity <= (i + 1) * segment_width:
                return sentiment_list[i]
        else:
            if polarity >= i * segment_width and polarity < (i + 1) * segment_width:
                return sentiment_list[i]


# *** PATHS ***
script_dir=os.path.dirname(__file__)
data_path = os.path.join(script_dir, '../data/processed/All_Scores_Without_Outliers.csv')
destination_path = os.path.join(script_dir, '../data/processed/With_ML_Sentiment.csv')

# preprocessor


# Dataframe
df = pd.read_csv(data_path)

# Perform sentiment analysis on each tweet
# Columns = [id,date,tweet,language,nlikes,nreplies,nretweets,Emoji Free,Document,Preprocessed Document,Score1,Score2,Overall_score]
sentiments = []
for tweet in df['tweet']:
    analysis = TextBlob(tweet)
    # You can choose to use polarity or subjectivity, or both
    sentiments.append(analysis.sentiment.polarity)

# Adding Polarity Score to the DataFrame
df['ml_polarity'] = sentiments

# Normalizing polarity scores to range: 0 to 1
scaler = MinMaxScaler(feature_range=(0, 1))
df['normalized_ml_polarity'] = scaler.fit_transform(df['ml_polarity'].values.reshape(-1, 1))

# *** ASSIGN SENTIMENT ***
sentiment_list = ['Negative', 'Neutral', 'Positive']
num_segments = 3

df['sentiment_s1'] = df['Score1_new'].apply(lambda polarity: assign_sentiment(polarity, sentiment_list, num_segments))
df['ml_sentiment'] = df['normalized_ml_polarity'].apply(lambda polarity: assign_sentiment(polarity, sentiment_list, num_segments))

# Export
df.to_csv(destination_path, index=False)
print(f'status: Exported .csv file to {destination_path}')


