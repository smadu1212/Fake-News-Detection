import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from preprocessing import load_and_preprocess

# Models save කරන්න 'models' folder එක සදාගැනීම
os.makedirs("models", exist_ok=True)

DATA_PATH = "data/WELFake_Dataset.csv"

# 1. Load and Preprocess Data
print("1️⃣ Data preprocess වෙමින් පවතී...")
df = load_and_preprocess(DATA_PATH)

X = df['clean_text']
y = df['label']

# 2. Train-Test Split (80% Train, 20% Test)
print("\n2️⃣ Data split කරමින් පවතී (80% Train, 20% Test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. TF-IDF Vectorization
print("\n3️⃣ TF-IDF Feature Extraction සිදුකෙරේ...")
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Vectorizer එක save කිරීම
with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)
print("✅ TF-IDF Vectorizer එක 'models/tfidf_vectorizer.pkl' ලෙස Save විය.")

# 4. Logistic Regression Training (Member 01 Model)
print("\n4️⃣ Logistic Regression Model එක Train වෙමින් පවතී...")
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)

y_pred_lr = lr_model.predict(X_test_tfidf)
print("\n--- 📊 Logistic Regression Performance ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_lr):.4f}")
print(classification_report(y_test, y_pred_lr, target_names=['Real', 'Fake']))

# Model එක save කිරීම
with open("models/logistic_regression.pkl", "wb") as f:
    pickle.dump(lr_model, f)

# 5. Random Forest Training (Member 03 Model)
print("\n5️⃣ Random Forest Model එක Train වෙමින් පවතී (මෙයට විනාඩි කිහිපයක් ගතවිය හැක)...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_tfidf, y_train)

y_pred_rf = rf_model.predict(X_test_tfidf)
print("\n--- 📊 Random Forest Performance ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_rf):.4f}")
print(classification_report(y_test, y_pred_rf, target_names=['Real', 'Fake']))

# Model එක save කිරීම
with open("models/random_forest.pkl", "wb") as f:
    pickle.dump(rf_model, f)

print("\n🎉 සියලුම Models සාර්ථකව Train වී Save විය!")