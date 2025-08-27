


#addimport streamlit as st
import pandas as pd
import joblib
import zipfile
import os
from bertopic import BERTopic
from utils import load_model, load_data, summarize_reviews

# ---- Page Config ----
st.set_page_config(
    page_title="Customer Reviews Insights",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---- Custom CSS for elegant black & white theme ----
st.markdown("""
    <style>
        body {
            background-color: #0e0e0e;
            color: #ffffff;
            font-family: 'Helvetica Neue', sans-serif;
        }
        .stApp {
            background-color: #0e0e0e;
        }
        .block-container {
            padding: 2rem 4rem;
        }
        h1, h2, h3, h4 {
            color: #f0f0f0;
        }
        .css-1d391kg { background-color: #1c1c1c; }
        .stTextInput input {
            background-color: #1c1c1c;
            color: white;
            border: 1px solid #444;
            border-radius: 10px;
        }
        .stButton>button {
            background-color: #1c1c1c;
            color: white;
            border-radius: 10px;
            border: 1px solid #555;
        }
        .stDataFrame, .stMarkdown {
            background-color: #141414;
            border-radius: 10px;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ---- Title ----
st.title(" Customer Reviews Insights")
st.write("Elegant **black & white dashboard** for Topic Modeling + Summarization")

# ---- Load model + data ----
model = load_model()
df = load_data()

# ---- User query input ----
query = st.text_input("🔍 Search topics related to:")
if query:
    similar_topics, similarity = model.find_topics(query)
    st.write("## Closest Topics")
    for t, s in zip(similar_topics, similarity):
        st.markdown(f"**Topic {t}** (score={s:.2f})")
        st.write(model.get_topic(t))

# ---- Summarization ----
if st.button(" Summarize All Reviews"):
    with st.spinner("Generating summary..."):
        summary = summarize_reviews(df["processed_text"].tolist())

    st.subheader("📝 Summary of Reviews")
    st.write(summary)
    
#added app.py
