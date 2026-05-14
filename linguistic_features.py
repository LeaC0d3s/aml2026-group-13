import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import textstat
import re

# loading splitted data
train_df = pd.read_csv("datasets/train.csv")
dev_df   = pd.read_csv("datasets/dev.csv")
test_df  = pd.read_csv("datasets/test.csv")

# cleaning text (removing URLs and extra whitespace)
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

for df in [train_df, dev_df, test_df]:
    df["text_clean"] = df["text"].apply(clean_text)

# sentiment feature with TextBlob (-1.0 negative → +1.0 positive )
def get_textblob_sentiment(text):
    return TextBlob(text).sentiment.polarity 

# sentiment feature with VADER (idem as above)
vader = SentimentIntensityAnalyzer()

def get_vader_sentiment(text):
    return vader.polarity_scores(text)["compound"] 

# feature gunning fog index
def get_gunning_fog(text):
    try:
        return textstat.gunning_fog(text)
    except Exception:
        return np.nan

# feature lexical diversity (usually low diversity means higher chance of being a fake news)
def get_lexical_diversity(text):
    tokens = text.lower().split()
    if len(tokens) == 0:
        return 0.0
    return len(set(tokens)) / len(tokens) # it is a list of words used and compared to the total number of words

# apply feature to dataset and add new columns with the features extracted
def extract_features(df):
    df["sentiment_textblob"] = df["text_clean"].apply(get_textblob_sentiment)
    df["sentiment_vader"]    = df["text_clean"].apply(get_vader_sentiment)
    df["gunning_fog"]        = df["text_clean"].apply(get_gunning_fog)
    df["lexical_diversity"]  = df["text_clean"].apply(get_lexical_diversity)
    return df

train_df = extract_features(train_df)
dev_df   = extract_features(dev_df)
test_df  = extract_features(test_df)

# Sanity check counts null and distribution of new features
feature_cols = ["sentiment_textblob", "sentiment_vader", "gunning_fog", "lexical_diversity"]

print(train_df[feature_cols + ["label"]].describe())
print("\nNull counts:", train_df[feature_cols].isnull().sum().to_dict())

# Save new dataframes with features
train_df.to_csv("datasets/train_features.csv", index=False)
dev_df.to_csv("datasets/dev_features.csv",     index=False)
test_df.to_csv("datasets/test_features.csv",   index=False)

print("\nSaved → train_features.csv, dev_features.csv, test_features.csv")