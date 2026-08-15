import os
import pickle
import numpy as np
import streamlit as st
from preprocessing import clean_text

# Page Configuration
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="centered"
)

# Custom Styling
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #1E88E5;
    }
    .sub-title {
        text-align: center;
        color: #555;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>📰 Fake News Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>NLP Project - Group 33 (Word Play)</p>", unsafe_allow_html=True)

# Load Saved Vectorizer and Model
@st.cache_resource
def load_resources():
    with open("models/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    
    with open("models/logistic_regression.pkl", "rb") as f:
        lr_model = pickle.load(f)
        
    return vectorizer, lr_model

try:
    vectorizer, lr_model = load_resources()
    st.success("✅ Model & Vectorizer Loaded Successfully!")
except Exception as e:
    st.error("❌ Models load කරගැනීමට නොහැකි විය.")
    st.stop()

# User Input Text Area
user_input = st.text_area(
    "📝 පුවත හෝ ලිපිය මෙතැනට ඇතුළත් කරන්න (News Text):",
    height=200,
    placeholder="Paste news headline or full text article here..."
)

# Predict Button
if st.button("🔍 Analyze Authenticity"):
    if not user_input.strip():
        st.warning("⚠️ කරුණාකර පරීක්ෂා කිරීමට text එකක් ඇතුළත් කරන්න.")
    else:
        with st.spinner("පුවත පරීක්ෂා කරමින් පවතී..."):
            # 1. Clean Text
            cleaned = clean_text(user_input)
            
            # 2. Vectorize
            text_vector = vectorizer.transform([cleaned])
            
            # 3. Model Prediction Probabilities
            probabilities = lr_model.predict_proba(text_vector)[0]
            
            # WELFake Dataset: Index 0 = Fake News, Index 1 = Real News
            prob_fake = probabilities[0]
            prob_real = probabilities[1]

            st.markdown("---")
            st.subheader("📊 Analysis Results")
            
            # Decision based on higher probability
            if prob_fake > prob_real:
                st.error("🔴 **Prediction:** FAKE NEWS")
                st.metric(label="Confidence Score", value=f"{prob_fake * 100:.2f}%")
            else:
                st.success("🟢 **Prediction:** REAL NEWS")
                st.metric(label="Confidence Score", value=f"{prob_real * 100:.2f}%")

            # Probability Breakdown Bars
            st.write("**Prediction Probabilities:**")
            st.progress(int(prob_real * 100), text=f"Real News Probability: {prob_real * 100:.2f}%")
            st.progress(int(prob_fake * 100), text=f"Fake News Probability: {prob_fake * 100:.2f}%")

# Disclaimer Footer
st.markdown("---")
st.caption("⚠️ **Disclaimer:** This AI model predicts news authenticity based on linguistic patterns. It does not perform real-time fact-checking.")