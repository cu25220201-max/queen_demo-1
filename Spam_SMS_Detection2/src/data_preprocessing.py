import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')
    stop_words = set(stopwords.words('english'))

RAW_DATA_PATH = "data/raw/spam.csv"
PROCESSED_DIR = "data/processed"
PROCESSED_FILE_PATH = os.path.join(PROCESSED_DIR, "clean_spam.csv")

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    words = word_tokenize(text)
    cleaned_words = [w for w in words if w not in stop_words]
    return " ".join(cleaned_words)

if __name__ == "__main__":
    print("==================================================")
    print("Loading Big Kaggle Dataset...")
    print("==================================================")
    
    if not os.path.exists(RAW_DATA_PATH):
        print(f"ERROR: File not found at {RAW_DATA_PATH}")
        exit()
        
   
    try:
        df = pd.read_csv(RAW_DATA_PATH, sep='\t', header=None, names=["label", "message"], encoding="utf-8")
    except:
       
        df = pd.read_csv(RAW_DATA_PATH, encoding="latin-1")
        df = df.iloc[:, :2]
        df.columns = ["label", "message"]
    
    print(f"Original Data Rows: {len(df)}")
    print("Cleaning text (Isme 10-15 seconds lag sakte hain)...")
    
    df['clean_message'] = df['message'].apply(clean_text)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(PROCESSED_FILE_PATH, index=False)
    
    print("==================================================")
    print("Big Dataset Processed Successfully!")
    print(f"Saved at: {PROCESSED_FILE_PATH}")
    print("==================================================")