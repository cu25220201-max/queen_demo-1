import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

PROCESSED_FILE_PATH = "data/processed/clean_spam.csv"
MODELS_DIR = "models"

if __name__ == "__main__":
    print("==================================================")
    print("Training AI Model...")
    print("==================================================")
    
    if not os.path.exists(PROCESSED_FILE_PATH):
        print("ERROR: Processed data nahi mila. Pehle data_preprocessing.py chalayein.")
        exit()
        
    df = pd.read_csv(PROCESSED_FILE_PATH)
    df.dropna(subset=['clean_message'], inplace=True)
    
    X = df['clean_message']
    y = df['label']
    
    
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(X)
    
   
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
    
    
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    
    preds = model.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, preds) * 100:.2f}%")
    
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(MODELS_DIR, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
        
    print("Model aur Vectorizer successfully save ho gaye 'models/' folder mein!")