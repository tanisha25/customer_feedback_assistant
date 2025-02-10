import openai
import faiss
import numpy as np
import pandas as pd

openai.api_key = "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"

import os
import numpy as np
import faiss
import openai

INDEX_PATH = "embeddings/vector_store.faiss"

def embed_text(text):
    """Generate OpenAI embeddings for a given text."""
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    return response["data"][0]["embedding"]

def store_embeddings(df, index_path=INDEX_PATH):
    """Convert text to embeddings and store in FAISS."""
    
    if "processed_comment" not in df.columns:
        print("⚠️ Column 'processed_comment' not found in DataFrame. Skipping embedding storage.")
        return
    
    df["processed_comment"] = df["processed_comment"].astype(str).fillna("")
    
    # Generate embeddings
    embeddings = np.array([embed_text(text) for text in df["processed_comment"]])

    # Initialize FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save the FAISS index
    faiss.write_index(index, index_path)
    print(f"✅ FAISS index saved at {index_path}")

def load_faiss_index(index_path=INDEX_PATH):
    """Load the FAISS vector store with error handling."""
    
    if not os.path.exists(index_path):
        print(f"⚠️ FAISS index file not found at {index_path}. Please generate embeddings first.")
        return None

    try:
        return faiss.read_index(index_path)
    except Exception as e:
        print(f"❌ Failed to load FAISS index: {e}")
        return None

