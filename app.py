import streamlit as st
import pandas as pd
from insights.data_preprocessing import preprocess_data
from insights.visualization import plot_top_words, analyze_sentiment
from chatbot.chatbot import generate_insights

# Preprocess data (if needed)
df = preprocess_data()

# Set up Streamlit app
st.set_page_config(page_title="📊 Customer Feedback Insights", layout="wide")

# Sidebar Navigation
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio("Select Page", ["Dashboard", "Chatbot"])

if page == "Dashboard":
    st.title("📊 Customer Review Insights for Redmi 6")

    st.subheader("Frequently Used Words")
    plot_top_words(df)

    st.subheader("Customer Sentiment Analysis")
    analyze_sentiment(df)


elif page == "Chatbot":
    st.title("💬 Customer Insights Chatbot")
    user_query = st.text_input("Ask about customer feedback:")
    if st.button("Get Insights"):
        response = generate_insights(user_query, df)
        st.write(response)

