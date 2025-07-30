# AI Inquiry Classifier
AI-powered customer inquiry classifier using Hugging Face with offline fallback.
This app classifies customer inquiries (e.g., refunds, complaints, feedback) using Hugging Face's zero-shot classifier, with a fallback to a local scikit-learn model if the API is unavailable.

## Features
- Uses Hugging Face API (bart-large-mnli)
- Falls back to local ML model if offline
- Built with Streamlit

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
