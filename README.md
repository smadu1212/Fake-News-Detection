\# 📰 Fake News Detection System (NLP)



An end-to-end Natural Language Processing (NLP) pipeline and real-time web application designed to classify news articles as \*\*REAL\*\* or \*\*FAKE\*\* using Machine Learning\[cite: 4, 5].



\---



\## 📌 Project Overview

\* \*\*Course:\*\* CCS3356 - Natural Language Processing\[cite: 3, 4]

\* \*\*Group:\*\* Group 33 (Word Play)\[cite: 3, 4]

\* \*\*Live Web App:\*\* \[fake-news-detection-lk.streamlit.app](https://fake-news-detection-lk.streamlit.app/)\[cite: 3, 4]

\* \*\*GitHub Repository:\*\* \[github.com/smadu1212/fake-news-detection](https://github.com/smadu1212/fake-news-detection)\[cite: 3, 4]



\### 👥 Team Members

| Student ID | Student Name | Assigned Role |

| :--- | :--- | :--- |

| \*\*CIT-24-01-0404\*\* | Lashan Pramuditha Dias | NLP Pipeline \& Data Preprocessing Lead\[cite: 5] |

| \*\*CIT-24-01-0081\*\* | H.M.S. Madusanka | Feature Vectorization \& Model Training Lead\[cite: 5] |

| \*\*CIT-24-01-0236\*\* | Yasitha Kalhara | Streamlit Cloud Deployment \& UI Lead\[cite: 5] |



\---



\## 🏗️ System Architecture \& NLP Pipeline



\* \*\*Text Preprocessing (`preprocessing.py`)\*\*: Case folding, regex noise/symbol stripping, NLTK stopword elimination\[cite: 4, 5].

\* \*\*Feature Extraction (`tfidf\_vectorizer.pkl`)\*\*: Scikit-learn TF-IDF Vectorizer with unigrams \& bigrams (`max\_features=5000`)\[cite: 4].

\* \*\*Classification Engine (`logistic\_regression.pkl`)\*\*: Regularized Logistic Regression with L-BFGS solver (`C=1.0`)\[cite: 4].

\* \*\*Live Web Interface (`app.py`)\*\*: Streamlit Cloud deployment with probability threshold evaluation and real-time confidence scores\[cite: 3, 4].



\---



\## 📁 Repository Structure

├── models/

│   ├── logistic\_regression.pkl

│   └── tfidf\_vectorizer.pkl

├── app.py

├── preprocessing.py

├── train.py

├── requirements.txt

└── README.md

