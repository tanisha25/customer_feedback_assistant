import pandas as pd
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("stopwords")
nltk.download("punkt")

def preprocess_text(text):
    """Clean and tokenize text data."""
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'\W', ' ', text)  # Remove special characters
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        return " ".join(tokens)
    return ""

def preprocess_data(input_path="data/redmi6.csv", output_path="data/processed_redmi6.csv"):
    """Preprocess the dataset if not already processed."""
    if os.path.exists(output_path):
        print("✅ Processed dataset already exists. Skipping preprocessing.")
        return pd.read_csv(output_path)  # Load preprocessed data

    print("⚙️ Preprocessing data...")
    df = pd.read_csv(input_path, encoding="ISO-8859-1")

    # Correctly map columns
    df.rename(columns={
        "Review Title": "review_title",
        "Customer": "customer_name",
        "Rating": "rating",
        "Date": "date",
        "Comments": "comments",
        "Useful": "usefulness"
    }, inplace=True)

    df["processed_comment"] = df["comments"].apply(preprocess_text)
    df.to_csv(output_path, index=False)
    print("✅ Data Preprocessing Completed!")
    return df  # Return the processed DataFrame
