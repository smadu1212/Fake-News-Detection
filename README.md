# 📰 Fake News Detection System (NLP)

An end-to-end Natural Language Processing (NLP) pipeline and real-time web application designed to classify news articles as **REAL** or **FAKE** using Machine Learning.

---

## 📌 Project Overview
* **Course:** CCS3356 - Natural Language Processing
* **Group:** Group 33 (Word Play)
* **Live Web App:** [fake-news-detection-lk.streamlit.app](https://fake-news-detection-lk.streamlit.app/)
* **GitHub Repository:** [github.com/smadu1212/fake-news-detection](https://github.com/smadu1212/fake-news-detection)

### 👥 Team Members
| Student ID | Student Name | Assigned Role |
| :--- | :--- | :--- |
| **CIT-24-01-0404** | Lashan Pramuditha Dias | NLP Pipeline & Data Preprocessing Lead |
| **CIT-24-01-0081** | H.M.S. Madusanka | Feature Vectorization & Model Training Lead |
| **CIT-24-01-0236** | Yasitha Kalhara | Streamlit Cloud Deployment & UI Lead |

---

## 🏗️ System Architecture & NLP Pipeline

* **Text Preprocessing (`preprocessing.py`)**: Case folding, regex noise/symbol stripping, NLTK stopword elimination.
* **Feature Extraction (`tfidf_vectorizer.pkl`)**: Scikit-learn TF-IDF Vectorizer with unigrams & bigrams (`max_features=5000`).
* **Classification Engine (`logistic_regression.pkl`)**: Regularized Logistic Regression with L-BFGS solver (`C=1.0`).
* **Live Web Interface (`app.py`)**: Streamlit Cloud deployment with probability threshold evaluation and real-time confidence scores.

---

## 📁 Repository Structure

