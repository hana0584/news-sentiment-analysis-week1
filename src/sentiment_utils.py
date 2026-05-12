from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np

def get_textblob_score(text):
    try:
        return TextBlob(str(text)).sentiment.polarity
    except:
        return 0.0

def get_vader_score(text):
    analyzer = SentimentIntensityAnalyzer()
    try:
        return analyzer.polarity_scores(str(text))['compound']
    except:
        return 0.0

def get_sentiment_label(score, threshold=0.05):
    if score > threshold:
        return 'Positive'
    elif score < -threshold:
        return 'Negative'
    else:
        return 'Neutral'

def align_news_to_trading_days(news_df, stock_df):
    """Align news dates to next available trading day."""
    news_df = news_df.copy()
    news_df['date'] = pd.to_datetime(news_df['date']).dt.tz_localize(None)
    # Get all unique trading dates from stock data
    trading_days = sorted(stock_df['Date'].unique())
    
    def get_next_trading_day(date):
        future = [d for d in trading_days if d >= date]
        return future[0] if future else pd.NaT
    
    news_df['aligned_date'] = news_df['date'].apply(get_next_trading_day)
    news_df = news_df.dropna(subset=['aligned_date'])
    return news_df

def aggregate_daily_sentiment(news_df):
    grouped = news_df.groupby(['stock', 'aligned_date'])
    daily = grouped.agg({
        'sentiment_vader': ['mean', 'std', 'count'],
        'sentiment_textblob': ['mean', 'std']
    }).round(4)
    daily.columns = ['_'.join(col).strip() for col in daily.columns.values]
    daily = daily.reset_index()
    daily['sentiment_category'] = daily['sentiment_vader_mean'].apply(
        lambda x: get_sentiment_label(x, threshold=0.05)
    )
    return daily 