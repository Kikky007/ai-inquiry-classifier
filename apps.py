# Create the full Streamlit app code with Hugging Face + scikit-learn fallback

import streamlit as st
import pandas as pd
import requests
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# --- CONFIG ---
LABELS = ["Complaint", "Refund Request", "Shipping Issue", "Product Information", "Positive Feedback", "Other"]
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
HEADERS = {"Authorization": f"Bearer {os.getenv('HF_API_KEY', '')}"}

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("inquiries.csv")
    return df

# --- Train Local Model ---
@st.cache_resource
def train_local_model(df):
    model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    model.fit(df['message'], df['category'])
    return model

# --- Predict with Hugging Face ---
def predict_with_huggingface(message):
    payload = {
        "inputs": message,
        "parameters": {"candidate_labels": LABELS},
    }
    response = requests.post(HUGGINGFACE_API_URL, headers=HEADERS, json=payload, timeout=10)
    if response.status_code == 200:
        result = response.json()
        return result["labels"][0]  # Top label
    else:
        raise RuntimeError(f"Hugging Face API failed: {response.status_code} - {response.text}")

# --- App UI ---
st.set_page_config(page_title="AI Inquiry Classifier", layout="centered")
st.title("AI Inquiry Classifier (Hugging Face + Local Fallback)")

df = load_data()
model = train_local_model(df)

user_input = st.text_area("Enter customer message:")

if st.button("Classify"):
    if not user_input.strip():
        st.warning("Please enter a message.")
    else:
        try:
            prediction = predict_with_huggingface(user_input)
            st.success(f"Hugging Face Prediction: {prediction}")
        except Exception as e:
            st.warning("Hugging Face API unavailable. Using local model instead.")
            prediction = model.predict([user_input])[0]
            st.success(f"Local Model Prediction: {prediction}")


