# import os
# import pandas as pd
# import gdown
# from bertopic import BERTopic
# from transformers import pipeline

# # ========================
# # 🔹 CONFIG: Your public file URLs
# # ========================
# CSV_URL = "https://drive.google.com/uc?id=1phcs2q0k7hPS2Lr27zDEJzW6HihrmkAd"
# MODEL_URL = "https://drive.google.com/uc?id=1b_dGde4-OlDqNK2XDDBvvJJ70gpuPTlb"  # not zip, direct folder/file

# # ---- Download helper ----
# def download_file(url, output):
#     if not os.path.exists(output):
#         print(f"Downloading {output} from Drive...")
#         gdown.download(url, output, quiet=False)
#         print(f"{output} downloaded.")

# # ---- Load BERTopic Model ----
# def load_model():
#     model_path = "bertopic_model"
#     if not os.path.exists(model_path):
#         # here we just download the model (no unzip step)
#         download_file(MODEL_URL, model_path)
#     model = BERTopic.load(model_path)
#     return model

# # ---- Load CSV ----
# def load_data():
#     if not os.path.exists("preprocessed_reviews.csv"):
#         download_file(CSV_URL, "preprocessed_reviews.csv")
#     return pd.read_csv("preprocessed_reviews.csv")

# # ---- Summarizer ----
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# def summarize_reviews(reviews, chunk_size=10):
#     summaries = []
#     for i in range(0, len(reviews), chunk_size):
#         chunk = " ".join(reviews[i:i+chunk_size])
#         if len(chunk.strip()) > 0:
#             summary = summarizer(chunk, max_length=120, min_length=40, do_sample=False)
#             summaries.append(summary[0]['summary_text'])
#     return " ".join(summaries)




import os
import zipfile
import pandas as pd
import gdown
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# =======================
# Configuration (update as needed)
# =======================
CSV_URL = "https://drive.google.com/uc?export=download&id=1phcs2q0k7hPS2Lr27zDEJzW6HihrmkAd"
MINILM_URL = "https://drive.google.com/uc?export=download&id=177ipf6srHDVmU1gxFoW77bWM_j_IK39j"

# ---- Download helper ----
def download_file(url, output_path):
    if not os.path.exists(output_path):
        print(f"Downloading {output_path} from Google Drive...")
        gdown.download(url, output_path, quiet=False)
        print(f"{output_path} downloaded.")

# ---- Load embedding model & BERTopic Model ----
def load_model():
    model_dir = "models/all-MiniLM-L6-v2"

    if not os.path.exists(model_dir):
        zip_file = "all-MiniLM-L6-v2.zip"
        download_file(MINILM_URL, zip_file)

        with zipfile.ZipFile(zip_file, "r") as z:
            z.extractall("models")

        print("MiniLM model extracted.")

    # Load the sentence-transformers model from local folder
    embedding_model = SentenceTransformer(model_dir)

    # Use it with BERTopic
    bertopic_model = BERTopic(embedding_model=embedding_model)

    return bertopic_model

# ---- Load CSV ----
def load_data():
    csv_path = "preprocessed_reviews.csv"
    if not os.path.exists(csv_path):
        download_file(CSV_URL, csv_path)
    return pd.read_csv(csv_path)

# ---- Summarizer ----
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_reviews(reviews, chunk_size=10):
    summaries = []
    for i in range(0, len(reviews), chunk_size):
        chunk = " ".join(reviews[i : i + chunk_size])
        if chunk.strip():
            summary = summarizer(chunk, max_length=120, min_length=40, do_sample=False)
            summaries.append(summary[0]["summary_text"])
    return " ".join(summaries)

