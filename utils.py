import os
import pandas as pd
import gdown
from bertopic import BERTopic
from transformers import pipeline

# ========================
# 🔹 CONFIG: Your public file URLs
# ========================
CSV_URL = "https://drive.google.com/uc?id=1phcs2q0k7hPS2Lr27zDEJzW6HihrmkAd"
MODEL_URL = "https://drive.google.com/uc?id=1b_dGde4-OlDqNK2XDDBvvJJ70gpuPTlb"  # not zip, direct folder/file

# ---- Download helper ----
def download_file(url, output):
    if not os.path.exists(output):
        print(f"Downloading {output} from Drive...")
        gdown.download(url, output, quiet=False)
        print(f"{output} downloaded.")

# ---- Load BERTopic Model ----
def load_model():
    model_path = "bertopic_model"
    if not os.path.exists(model_path):
        # here we just download the model (no unzip step)
        download_file(MODEL_URL, model_path)
    model = BERTopic.load(model_path)
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
