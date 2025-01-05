import streamlit as st
from googleapiclient.discovery import build
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO

# Download NLTK data
nltk.download("vader_lexicon")

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Function to fetch YouTube comments
def fetch_comments(api_key, video_id, max_comments=50):
    youtube = build("youtube", "v3", developerKey=api_key)
    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            pageToken=next_page_token,
            maxResults=min(100, max_comments - len(comments))
        ).execute()

        for item in response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return comments

# Function to fetch YouTube video details
def fetch_video_details(api_key, video_id):
    youtube = build("youtube", "v3", developerKey=api_key)
    response = youtube.videos().list(
        part="snippet",
        id=video_id
    ).execute()

    if "items" in response and len(response["items"]) > 0:
        title = response["items"][0]["snippet"]["title"]
        return title
    return None

# Function to perform sentiment analysis
def analyze_comments(comments):
    results = []
    for comment in comments:
        sentiment = sia.polarity_scores(comment)
        sentiment["comment"] = comment
        sentiment["label"] = (
            "Positive" if sentiment["compound"] > 0.2 else
            "Negative" if sentiment["compound"] < -0.2 else
            "Neutral"
        )
        results.append(sentiment)
    return pd.DataFrame(results)

# Function to fetch YouTube video thumbnail
def fetch_thumbnail(video_id):
    url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        return None

# Streamlit app
def main():
    st.title("YouTube Comment Sentiment Analysis")
    
    # API Key and Video ID Input (API Key specified directly in the code)
    api_key = "AIzaSyDY_v1QHMx3Me_lF-5I4ZtuV4vcRVPMfk0"  # Your API Key here
    video_url = st.sidebar.text_input("Enter YouTube Video URL")

    max_comments = st.sidebar.slider("Number of Comments to Analyze", 10, 200, 50)

    if st.sidebar.button("Analyze"):
        if not video_url:
            st.error("Please provide the YouTube Video URL.")
            return

        # Extract video ID from URL
        try:
            video_id = video_url.split("v=")[-1].split("&")[0]
        except IndexError:
            st.error("Invalid YouTube URL format.")
            return

        # Display Video Title
        st.info("Fetching video title...")
        video_title = fetch_video_details(api_key, video_id)
        if video_title:
            st.subheader(f"Video Title: {video_title}")
        else:
            st.warning("Failed to fetch video title.")

        # Display Video Thumbnail
        st.info("Fetching video thumbnail...")
        thumbnail = fetch_thumbnail(video_id)
        if thumbnail:
            st.image(thumbnail, caption="Video Thumbnail", use_column_width=True)
        else:
            st.warning("Failed to fetch video thumbnail.")

        st.info("Fetching comments...")
        try:
            comments = fetch_comments(api_key, video_id, max_comments)
        except Exception as e:
            st.error(f"Error fetching comments: {e}")
            return

        if not comments:
            st.warning("No comments found for this video.")
            return

        st.success(f"Fetched {len(comments)} comments.")
        
        # Sentiment Analysis
        st.info("Analyzing sentiment...")
        sentiment_df = analyze_comments(comments)

        # Display Sentiment Analysis Results
        st.header("Sentiment Distribution")
        sentiment_counts = sentiment_df["label"].value_counts()
        fig, ax = plt.subplots()
        sentiment_counts.plot(kind="bar", color=["green", "gray", "red"], ax=ax)
        ax.set_title("Sentiment Distribution")
        ax.set_ylabel("Number of Comments")
        ax.set_xticklabels(sentiment_counts.index, rotation=0)
        st.pyplot(fig)

        # Display Comments with Sentiment
        st.header("Comments with Sentiment")
        for _, row in sentiment_df.iterrows():
            sentiment_color = (
                "green" if row["label"] == "Positive" else
                "red" if row["label"] == "Negative" else
                "blue"
            )
            st.markdown(f"<p style='color:{sentiment_color};'><b>Comment:</b> {row['comment']}</p>", unsafe_allow_html=True)
            st.write(f"**Sentiment:** {row['label']}")
            st.write("---")

        # Downloadable CSV
        st.download_button(
            label="Download Sentiment Analysis as CSV",
            data=sentiment_df.to_csv(index=False),
            file_name="sentiment_analysis.csv",
            mime="text/csv",
        )

if __name__ == "__main__":
    main()
