import streamlit as st
from services.query_engine import generate_insights
import pandas as pd
from insights.data_preprocessing import preprocess_data

st.title("📊 Customer Feedback Chatbot")

# Preprocess data (if needed)
df = preprocess_data()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.text_input("Ask a question about customer feedback:")

if st.button("Get Insights"):
    if user_query:
        response = generate_insights(user_query, df)
        st.session_state.chat_history.append(("User", user_query))
        st.session_state.chat_history.append(("Assistant", response))

        for role, text in st.session_state.chat_history:
            st.write(f"**{role}**: {text}")
