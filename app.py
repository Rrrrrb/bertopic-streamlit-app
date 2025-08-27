import streamlit as st
import pandas as pd
import joblib
import zipfile
import os
from bertopic import BERTopic

# ---- Custom Styling ----
def set_custom_style():
    st.markdown(
        """
        <style>
        /* Background */
        .stApp {
            background-color: #0f0f0f;
            color: #f5f5f5;
            font-family: 'Helvetica Neue', sans-serif;
        }

        /* Titles */
        h1, h2, h3 {
            color: white !important;
            font-weight: 600;
            letter-spacing: 1px;
        }

        /* Input box */
        .stTextInput>div>div>input {
            background-color: #1a1a1a;
            color: white;
            border: 1px solid #444;
            border-radius: 8px;
            padding: 10px;
        }

        /* Boxes (results) */
        .result-box {
            background-color: #1a1a1a;
            padding: 15px;
            margin-top: 15px;
            border-radius: 10px;
            border: 1px solid #333;
        }

        /* Buttons */
        .stButton>button {
            background-color: #ffffff;
            color: black;
            font-weight: 600;
            border-radius: 8px;
            border: none;
            padding: 8px 16px;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #e6e6e6;
            transform: translateY(-2px);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# ---- Load model ----
@st.cache_resource
def load_model():
    if not os.path.exists("bertopic_model"):
        with zipfile.ZipFile("bertopic_model.zip", "r") as zip_ref:
            zip_ref.extractall("bertopic_model")
    model = BERTopic.load("bertopic_model")
    return model

# ---- Load data ----
@st.cache_data
def load_data():
    return pd.read_csv("preprocessed_reviews.csv")

# ---- App ----
set_custom_style()
st.title("🖤 Customer Reviews Topic Explorer")
st.write("A **minimal & elegant** topic discovery tool powered by **BERTopic**")

# Load model + data
model = load_model()
df = load_data()

# ---- User input ----
query = st.text_input("Search topics related to:")
if query:
    similar_topics, similarity = model.find_topics(query)
    st.write("### 🔍 Closest Topics")
    for t, s in zip(similar_topics, similarity):
        with st.container():
            st.markdown(
                f"""
                <div class="result-box">
                    <p><b>Topic {t}</b> (score={s:.2f})</p>
                    <p>{model.get_topic(t)}</p>
                </div>
                """,
                unsafe_allow_html=True
            )



#added app.py
