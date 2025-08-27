import pandas as pd
import os, zipfile, requests
from bertopic import BERTopic
from transformers import pipeline

# ========================
# 🔹 CONFIG: Your public file URLs
# ========================
CSV_URL = "https://drive.google.com/uc?export=download&id=1phcs2q0k7hPS2Lr27zDEJzW6HihrmkAd"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1b_dGde4-OlDqNK2XDDBvvJJ70gpuPTlb"


# ---- Download helper ----
def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        r = requests.get(url, stream=True)
        with open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"{filename} downloaded.")

# ---- Load BERTopic Model ----
def load_model():
    if not os.path.exists("bertopic_model"):
        download_file(MODEL_URL, "bertopic_model.zip")
        with zipfile.ZipFile("bertopic_model.zip", "r") as zip_ref:
            zip_ref.extractall("bertopic_model")
    model = BERTopic.load("bertopic_model")
    return model

# ---- Load CSV ----
def load_data():
    if not os.path.exists("preprocessed_reviews.csv"):
        download_file(CSV_URL, "preprocessed_reviews.csv")
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

