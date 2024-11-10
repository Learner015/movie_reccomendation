import pandas as pd
import re
from textblob import TextBlob

# Load the dataset
rec = pd.read_csv('./movies.csv', encoding='ISO-8859-1')
print(rec.head())

# Check column names
print(rec.columns)

# Define the function to clean the data
def clean_data(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\S+', '', text)     # Remove mentions
    text = re.sub(r'#', '', text)        # Remove hashtags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()                  # Convert to lowercase
    return text

# Apply the cleaning function to the 'text_data' column
rec['cleaned_text'] = rec['genres'].apply(clean_data)
print(rec['cleaned_text'].head())

# Define the function to get sentiment
def get_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    return 'positive' if polarity > 0 else ('negative' if polarity < 0 else 'neutral')

# Apply the sentiment analysis function to the 'cleaned_text' column
rec['sentiment'] = rec['cleaned_text'].apply(get_sentiment)
print(rec[['cleaned_text', 'sentiment']].head())

import matplotlib.pyplot as plt # Visualize sentiment distribution 
sentiment_counts = rec['sentiment'].value_counts() 
sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue']) 
plt.title('Sentiment Distribution') 
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

