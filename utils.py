import pandas as pd
import os, zipfile
from bertopic import BERTopic
from transformers import pipeline

# ---- Load BERTopic Model ----
def load_model():
    if not os.path.exists("bertopic_model"):
        with zipfile.ZipFile("bertopic_model.zip", "r") as zip_ref:
            zip_ref.extractall("bertopic_model")
    model = BERTopic.load("bertopic_model")
    return model

# ---- Load CSV ----
def load_data():
    return pd.read_csv("preprocessed_reviews.csv")

# ---- Summarizer ----
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_reviews(reviews, chunk_size=10):
    summaries = []
    for i in range(0, len(reviews), chunk_size):
        chunk = " ".join(reviews[i:i+chunk_size])
        if len(chunk.strip()) > 0:
            summary = summarizer(chunk, max_length=120, min_length=40, do_sample=False)
            summaries.append(summary[0]['summary_text'])
    return " ".join(summaries)
