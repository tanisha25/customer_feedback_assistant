import openai
import faiss
import numpy as np
import pandas as pd
from services.vector_store import embed_text, load_faiss_index

openai.api_key = "<OPEN_API_KEY>"


def retrieve_relevant_docs(query, df, top_k=5):
    """Retrieve the most relevant feedback based on query embeddings."""
    
    index = load_faiss_index()
    
    if index is None:
        print("⚠️ FAISS index is not available. Ensure embeddings are created.")
        return df.head(top_k)  # Return the top K rows as a fallback

    query_embedding = np.array(embed_text(query)).reshape(1, -1)
    
    distances, indices = index.search(query_embedding, top_k)
    
    return df.iloc[indices[0]]

def generate_insights(query, df):
    """Use GPT-4 to generate insights from retrieved customer reviews."""
    
    retrieved_docs = retrieve_relevant_docs(query, df)
    
    if retrieved_docs.empty:
        return "No relevant insights found."
    
    feedback_text = "\n".join(retrieved_docs["processed_comment"].tolist())

    prompt = f"""
    You are a customer experience analyst. Based on the following customer reviews, 
    provide structured insights and key improvement areas.

    Reviews:
    {feedback_text}

    Key Issues:
    Customer Sentiment:
    Actionable Recommendations:
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}]
    )

    return response["choices"][0]["message"]["content"]
