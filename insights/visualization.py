import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
import numpy as np
import streamlit as st  # ✅ Import Streamlit

def plot_top_words(df, num_words=10):
    """Show most frequently mentioned words as a horizontal bar chart."""

    if "processed_comment" not in df.columns:
        st.warning("⚠️ Column 'processed_comment' not found in DataFrame. Skipping visualization.")
        return

    # Convert all comments to string and handle NaN values
    df["processed_comment"] = df["processed_comment"].astype(str).fillna("")

    # Count word frequencies
    word_counts = pd.Series(" ".join(df["processed_comment"]).split()).value_counts().head(num_words)

    # Plot the bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=word_counts.values, y=word_counts.index, palette="viridis", ax=ax)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Words")
    ax.set_title(f"🔝 Top {num_words} Most Frequent Words in Reviews")

    # ✅ Display in Streamlit
    st.pyplot(fig)

def analyze_sentiment(df):
    """Analyze sentiment distribution (positive, neutral, negative)."""

    if "processed_comment" not in df.columns:
        st.warning("⚠️ Column 'processed_comment' not found in DataFrame. Skipping sentiment analysis.")
        return

    # Convert all comments to string and handle NaN values
    df["processed_comment"] = df["processed_comment"].astype(str).fillna("")

    # Compute polarity scores
    df["polarity"] = df["processed_comment"].apply(lambda x: TextBlob(x).sentiment.polarity)

    # Define sentiment categories
    conditions = [
        df["polarity"] > 0,   # Positive
        df["polarity"] == 0,  # Neutral
        df["polarity"] < 0    # Negative
    ]
    labels = ["Positive", "Neutral", "Negative"]

    # Apply sentiment labels using np.select (Fix KeyError)
    df["sentiment"] = np.select(conditions, labels, default="Neutral")

    # Plot sentiment distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    df["sentiment"].value_counts().plot.pie(autopct="%1.1f%%", colors=["green", "grey", "red"], ax=ax)
    ax.set_title("📊 Sentiment Distribution")
    ax.set_ylabel("")  # Hide y-label

    # ✅ Display in Streamlit
    st.pyplot(fig)

