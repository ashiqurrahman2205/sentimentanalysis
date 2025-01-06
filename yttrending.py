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

# Function to fetch trending YouTube videos
def fetch_trending_videos(api_key, region_code="US", max_results=10):
    youtube = build("youtube", "v3", developerKey=api_key)
    response = youtube.videos().list(
        part="snippet",
        chart="mostPopular",
        regionCode=region_code,
        maxResults=max_results
    ).execute()

    videos = []
    for item in response.get("items", []):
        videos.append({
            "video_id": item["id"],
            "title": item["snippet"]["title"],
            "thumbnail_url": item["snippet"]["thumbnails"]["high"]["url"]
        })
    return videos

# Function to fetch YouTube video details
def fetch_video_details(api_key, video_id):
    youtube = build("youtube", "v3", developerKey=api_key)
    response = youtube.videos().list(
        part="snippet",
        id=video_id
    ).execute()

    if "items" in response and len(response["items"]) > 0:
        return response["items"][0]["snippet"]["title"]
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

# YouTube Comment Analysis page
def youtube_comment_analysis_page(api_key):
    st.sidebar.subheader("Input Options")
    video_source = st.sidebar.radio("Select Source", ["Trending Videos", "Video URL"])

    if video_source == "Trending Videos":
        region_code = st.sidebar.text_input("Region Code (e.g., US, IN)", value="US")
        st.sidebar.info("Fetching trending videos...")
        trending_videos = fetch_trending_videos(api_key, region_code)

        if trending_videos:
            st.subheader("Trending Videos")
            for video in trending_videos:
                st.image(video["thumbnail_url"], width=200)
                st.write(f"**{video['title']}**")
                if st.button(f"Analyze Comments for {video['title']}"):
                    video_id = video["video_id"]
                    analyze_video(api_key, video_id)
        else:
            st.warning("No trending videos found.")
    elif video_source == "Video URL":
        video_url = st.sidebar.text_input("Enter YouTube Video URL")
        if st.sidebar.button("Analyze"):
            try:
                video_id = video_url.split("v=")[-1].split("&")[0]
                analyze_video(api_key, video_id)
            except IndexError:
                st.error("Invalid YouTube URL format.")

# Analyze video and display sentiment analysis
def analyze_video(api_key, video_id):
    st.info("Fetching video title...")
    video_title = fetch_video_details(api_key, video_id)
    if video_title:
        st.subheader(f"Video Title: {video_title}")
    else:
        st.warning("Failed to fetch video title.")

    st.info("Fetching comments...")
    try:
        comments = fetch_comments(api_key, video_id)
    except Exception as e:
        st.error(f"Error fetching comments: {e}")
        return

    if not comments:
        st.warning("No comments found for this video.")
        return

    st.success(f"Fetched {len(comments)} comments.")

    st.info("Analyzing sentiment...")
    sentiment_df = analyze_comments(comments)

    # Sentiment Distribution
    st.header("Sentiment Distribution")
    sentiment_counts = sentiment_df["label"].value_counts()
    fig, ax = plt.subplots()
    sentiment_counts.plot(kind="bar", color=["green", "gray", "red"], ax=ax)
    ax.set_title("Sentiment Distribution")
    ax.set_ylabel("Number of Comments")
    ax.set_xticklabels(sentiment_counts.index, rotation=0)
    st.pyplot(fig)

    # Comments with Sentiment
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

# Main Streamlit app
def main():
    st.title("YouTube Comment Sentiment Analysis App")

    api_key = "AIzaSyDY_v1QHMx3Me_lF-5I4ZtuV4vcRVPMfk0"  # Your API Key here

    youtube_comment_analysis_page(api_key)

if __name__ == "__main__":
    main()
