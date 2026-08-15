import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 1. NLTK Data Download කරගැනීම (පළමු වරට රන් වන විට අවශ්‍ය වේ)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Text cleaning function:
    - Lowercasing (සියලු අකුරු small letters බවට පත් කිරීම)
    - HTML tags & URLs ඉවත් කිරීම
    - Punctuation සහ Numbers ඉවත් කිරීම
    - Stopwords ඉවත් කිරීම සහ Lemmatization (වචන මුල් ස්වරූපයට ගෙන ඒම)
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercasing
    text = text.lower()
    
    # 2. HTML tags ඉවත් කිරීම
    text = re.sub(r'<.*?>', '', text)
    
    # 3. URLs ඉවත් කිරීම
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 4. Special characters, punctuation & numbers ඉවත් කිරීම
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 5. Tokenization, Stopwords Removal & Lemmatization
    words = text.split()
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return " ".join(cleaned_words)

def load_and_preprocess(file_path):
    """
    CSV File එක load කර Preprocessing සිදු කරන ප්‍රධාන function එක.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File එක සොයාගත නොහැකි විය: {file_path}")
        
    print(f"📥 Dataset එක Load වෙමින් පවතී: {file_path}")
    df = pd.read_csv(file_path)
    
    # Missing values පිරවීම
    df['title'] = df['title'].fillna('')
    df['text'] = df['text'].fillna('')
    
    # Title එකයි Text එකයි එකතු කර full text එකක් සදාගැනීම
    df['full_text'] = df['title'] + " " + df['text']
    
    print("🧹 Text Cleaning ආරම්භ විය (මෙයට විනාඩි කිහිපයක් ගතවිය හැක)...")
    df['clean_text'] = df['full_text'].apply(clean_text)
    
    # හිස් වූ rows ඉවත් කිරීම
    df = df[df['clean_text'].str.strip() != '']
    
    print("✅ Preprocessing සාර්ථකව අවසන් විය!")
    return df

if __name__ == "__main__":
    # Test කර බලමු
    DATA_PATH = "data/WELFake_Dataset.csv"
    
    try:
        cleaned_df = load_and_preprocess(DATA_PATH)
        print("\n--- Cleaned Data Sample ---")
        print(cleaned_df[['full_text', 'clean_text', 'label']].head())
    except Exception as e:
        print(e)