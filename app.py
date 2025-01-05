# %% [markdown]
# Streamlit Application for Subreddit Sentiment Analysis

# %%
import streamlit as st
import praw
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords

sns.set(style='darkgrid', context='talk', palette='Dark2')

# Download necessary NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

# Function to fetch subreddit posts
def fetch_subreddit_posts(subreddit_name, limit=100):
    reddit = praw.Reddit(
        client_id='6uiEVMTD9IexgyD_Fd8fvw',
        client_secret='_Kpy5-BXm1CV5sQUbLJ1QLBjiGL8BQ',
        user_agent='ashiqurrahman2205'
    )
    headlines = []
    for submission in reddit.subreddit(subreddit_name).new(limit=limit):
        headlines.append(submission.title)
    return headlines

# Function to perform sentiment analysis
def analyze_sentiment(headlines):
    sia = SIA()
    results = []
    for line in headlines:
        pol_score = sia.polarity_scores(line)
        pol_score['headline'] = line
        results.append(pol_score)
    df = pd.DataFrame.from_records(results)
    df['label'] = 0
    df.loc[df['compound'] > 0.2, 'label'] = 1
    df.loc[df['compound'] < -0.2, 'label'] = -1
    return df

# Streamlit app starts here
st.title("Subreddit Sentiment Analysis")
st.sidebar.header("Configuration")

# User inputs
subreddit_name = st.sidebar.text_input("Enter Subreddit Name", "politics")
post_limit = st.sidebar.slider("Number of Posts to Analyze", 10, 500, 100)

# Fetch and analyze data
if st.sidebar.button("Analyze"):
    with st.spinner(f"Fetching posts from r/{subreddit_name}..."):
        headlines = fetch_subreddit_posts(subreddit_name, limit=post_limit)
    
    st.success(f"Fetched {len(headlines)} posts from r/{subreddit_name}.")

    with st.spinner("Analyzing sentiment..."):
        df = analyze_sentiment(headlines)
    
    st.success("Sentiment analysis completed.")
    
    # Display results
    st.header("Sentiment Distribution")
    counts = df['label'].value_counts(normalize=True) * 100
    fig, ax = plt.subplots()
    sns.barplot(x=counts.index, y=counts, ax=ax)
    ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
    ax.set_ylabel("Percentage")
    st.pyplot(fig)

    st.header("Sample Headlines")
    st.subheader("Positive Headlines")
    st.write(df[df['label'] == 1].headline.head(5).to_list())
    
    st.subheader("Negative Headlines")
    st.write(df[df['label'] == -1].headline.head(5).to_list())

    # Save results to a CSV file
    df.to_csv(f'{subreddit_name}_sentiment_analysis.csv', index=False)
    st.download_button(
        label="Download Sentiment Analysis CSV",
        data=df.to_csv(index=False),
        file_name=f'{subreddit_name}_sentiment_analysis.csv',
        mime='text/csv'
    )

st.sidebar.markdown("Developed by Watt Warriors ⚡")
