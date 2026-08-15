import os
import pickle
import streamlit as st
import numpy as np
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

# Model Selection
model_choice = st.selectbox(
    "🎯 භාවිතා කිරීමට අවශ්‍ය Model එක තෝරන්න:",
    ("Logistic Regression", "Random Forest Classifier")
)

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
            
            # 3. Model Prediction
            selected_model = lr_model if "Logistic Regression" in model_choice else rf_model
            prediction = selected_model.predict(text_vector)[0]
            probabilities = selected_model.predict_proba(text_vector)[0]
            
            confidence = np.max(probabilities) * 100

            st.markdown("---")
            st.subheader("📊 Analysis Results")
            
           # Label Output (WELFake Dataset: 0 = Fake, 1 = Real)
            if prediction == 0:
                st.error(f"🔴 **Prediction:** FAKE NEWS")
                st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
            else:
                st.success(f"🟢 **Prediction:** REAL NEWS")
                st.metric(label="Confidence Score", value=f"{confidence:.2f}%")

            # Probability Breakdown Bar
            st.write("**Prediction Probabilities:**")
            st.progress(int(probabilities[0] * 100), text=f"Fake News Probability: {probabilities[0]*100:.2f}%")
            st.progress(int(probabilities[1] * 100), text=f"Real News Probability: {probabilities[1]*100:.2f}%")
# Disclaimer Footer
st.markdown("---")
st.caption("⚠️ **Disclaimer:** This AI model predicts news authenticity based on linguistic patterns[cite: 1]. It does not perform real-time fact-checking[cite: 1].")