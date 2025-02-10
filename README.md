
# 📊 Customer Feedback Chatbot

This project is designed to analyze and provide insights into customer feedback using NLP techniques and OpenAI's GPT-4. The app allows users to query a dataset of customer reviews, get insights about customer sentiment, and visualize frequently mentioned words and sentiment distribution.

---

## Table of Contents
- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Setup Instructions](#setup-instructions)
- [Features](#features)
  - [Preprocessing](#preprocessing)
  - [Word Cloud & Sentiment Analysis](#word-cloud--sentiment-analysis)
  - [Chatbot Insights](#chatbot-insights)
- [Usage](#usage)
- [Visualizations](#visualizations)
  - [Top Words](#top-words)
  - [Sentiment Distribution](#sentiment-distribution)
- [Contributors](#contributors)
- [License](#license)

---

## Overview

This project utilizes a customer review dataset to build a Streamlit application that offers:
- Text preprocessing (removing special characters, stopwords, and tokenization)
- Visualization of frequently mentioned words in customer reviews
- Sentiment analysis (positive, neutral, and negative sentiments)
- A chatbot for querying customer feedback and generating actionable insights using GPT-4

---

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: OpenAI GPT-4
- **Libraries**: 
  - `pandas`, `nltk`, `seaborn`, `matplotlib`, `textblob`
  - `faiss` (for similarity search with embeddings)
- **APIs**: OpenAI API for generating insights and embeddings
- **Visualization**: WordCloud, bar charts, pie charts

---

## Setup Instructions

### Prerequisites

1. Install Python 3.x.
2. Install the required libraries using pip:

```bash
pip install -r requirements.txt
```

3. Get an API key from OpenAI and set it up in your environment.

### File Structure

```
customer-feedback-chatbot/
│
├── chatbot/
│   └── chatbot.py                # Main chatbot interface
│
├── services/
│   ├── query_engine.py           # Query engine for insights generation
│   └── vector_store.py           # Handling embeddings and FAISS
│
├── insights/
│   ├── data_preprocessing.py    # Data cleaning and preprocessing
│   ├── visualization.py         # Visualization functions
│
├── data/
│   ├── redmi6.csv               # Raw data
│   └── processed_redmi6.csv     # Processed data
│
├── app.py                       # Main Streamlit app
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

---

## Features

### Preprocessing

The dataset is preprocessed by cleaning the text (removing special characters and stopwords), tokenizing it, and generating embeddings for similarity-based queries.

### Word Cloud & Sentiment Analysis

- **Word Cloud**: Displays the most frequently mentioned words in the reviews.
- **Sentiment Analysis**: Categorizes reviews as positive, neutral, or negative based on their polarity.

### Chatbot Insights

The chatbot lets users ask questions about the reviews. It retrieves relevant customer feedback based on query embeddings and generates insights using GPT-4.

---

## Usage

### Running the Application

To run the app, simply use:

```bash
streamlit run app.py
```

This will open the Streamlit interface in your browser, where you can navigate between the **Dashboard** and **Chatbot** views.

### Chatbot Interaction

- Ask questions like: 
  - "What are the common complaints from customers?"
  - "How do customers feel about the Redmi 6?"
  - "What improvements do customers suggest?"
- The chatbot will generate insights and actionable recommendations based on the reviews.

---
# Customer Feedback Assistant

This project aims to build a **RAG-enabled assistant** that processes customer feedback and provides insights using machine learning models.

## Key Features

- Natural Language Processing for analyzing feedback
- Sentiment analysis to identify positive and negative reviews
- Insights extraction to identify top frequent words and patterns

## Visualizations

### Top 10 Frequent Words in Reviews

![Top Words Visualization](images/Top 10 Frequent Words in Reviews.png)

Figure 1: Top 10 Words from Customer Reviews

This visualization shows the most common words mentioned in the customer reviews.

### Sentiment Distribution

![Sentiment Distribution](images/Sentiment Distribution.png)

Figure 2: Customer Sentiment Distribution

This pie chart displays the distribution of customer sentiments (Positive, Neutral, Negative).

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/tanisha25/customer_feedback_assistant.git
