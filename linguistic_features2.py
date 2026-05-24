import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import textstat
import re

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # made lowercase to uniform fake and real news 
    text = text.lower()

    # fixing space-quote anomalies (" ) or loose quotes
    text = re.sub(r'^"\s+', '', text) 
    text = re.sub(r'["\'”“’‘]', ' ', text)

    # drop certain headline flags (Factbox:, Timeline:)
    text = re.sub(r"^(factbox|timeline|highlights|exclusive|correcting|update\s*\d*)\s*:\s*", "", text)
    
    # strip everything up to the first dash if it mentions reuters
    if "reuters" in text:
        text = re.sub(r"^.*?[\-—–]", "", text)
        
    # erase remaining giveaways
    text = re.sub(r"reuters", "", text)
    text = re.sub(r"\[.*?\]|\(.*?\)", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[!?.#@*]", " ", text)
    text = re.sub(r"[\-—–_]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

# sentiment feature with TextBlob (-1.0 negative → +1.0 positive)
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
    tokens = text.split()
    if len(tokens) == 0:
        return 0.0
    return len(set(tokens)) / len(tokens)

def run_master_clean():
    # loading splitted data
    train_df = pd.read_csv("datasets/train.csv")
    dev_df   = pd.read_csv("datasets/dev.csv")
    test_df  = pd.read_csv("datasets/test.csv")
    
    # cleaning text and title
    for df in [train_df, dev_df, test_df]:
        if "title" in df.columns:
            df["title"] = df["title"].astype(str).apply(clean_text)
        if "text" in df.columns:
            df["text"] = df["text"].astype(str).apply(clean_text)
        df["text_clean"] = df["text"]
        
    # apply feature to dataset and add new columns with the features extracted
    for df in [train_df, dev_df, test_df]:
        df["sentiment_textblob"] = df["text_clean"].apply(get_textblob_sentiment)
        df["sentiment_vader"]    = df["text_clean"].apply(get_vader_sentiment)
        df["gunning_fog"]        = df["text_clean"].apply(get_gunning_fog)
        df["lexical_diversity"]  = df["text_clean"].apply(get_lexical_diversity)
        df["gunning_fog"]        = df["gunning_fog"].fillna(0)
    
    # Save new dataframes with features
    train_df.to_csv("datasets/train_features.csv", index=False)
    dev_df.to_csv("datasets/dev_features.csv",     index=False)
    test_df.to_csv("datasets/test_features.csv",   index=False)
    print("\nSaved → train_features.csv, dev_features.csv, test_features.csv")
    
if __name__ == "__main__":
    run_master_clean()