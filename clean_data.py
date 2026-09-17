import duckdb
import pandas as pd
import re
from langdetect import detect, LangDetectException

BRAND = "AppleSupport"

con = duckdb.connect()
df = con.execute(f"""
    SELECT tweet_id, author_id, inbound, created_at, text,
           response_tweet_id, in_response_to_tweet_id
    FROM read_csv_auto('data/twcs.csv')
    WHERE author_id = '{BRAND}'
       OR tweet_id IN (
           SELECT in_response_to_tweet_id
           FROM read_csv_auto('data/twcs.csv')
           WHERE author_id = '{BRAND}' AND in_response_to_tweet_id IS NOT NULL
       )
""").df()

print(f"Loaded {len(df)} rows for {BRAND}")

customer_tweets = df[df['inbound'] == True]
brand_replies = df[df['inbound'] == False]

pairs = customer_tweets.merge(
    brand_replies,
    left_on='tweet_id',
    right_on='in_response_to_tweet_id',
    suffixes=('_customer', '_brand')
)

print(f"Matched {len(pairs)} customer-brand reply pairs")

# Keep only conversations that START with the customer — i.e. the customer's
# tweet wasn't itself a reply to an earlier tweet. This gives us the real
# initial complaint, not a mid-thread fragment answering a clarifying question.
pairs = pairs[pairs['in_response_to_tweet_id_customer'].isna()]

print(f"First-message-only pairs: {len(pairs)} rows")

def clean_text(text):
    text = str(text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'#+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

pairs['customer_text_clean'] = pairs['text_customer'].apply(clean_text)
pairs['brand_text_clean'] = pairs['text_brand'].apply(clean_text)

pairs = pairs[(pairs['customer_text_clean'] != '') & (pairs['brand_text_clean'] != '')]

def is_english(text):
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False

pairs = pairs[pairs['customer_text_clean'].apply(is_english)]

print(f"Final clean dataset: {len(pairs)} rows")

pairs.to_csv('data/clean_pairs.csv', index=False)
print("Saved to data/clean_pairs.csv")