# app/preprocessing.py
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

nltk.download('stopwords')
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """Remove punctuation, numbers, stopwords, lowercase"""
    text = re.sub(r"[^a-zA-Z]", " ", text)  # keep only letters
    text = text.lower()
    words = [word for word in text.split() if word not in STOPWORDS]
    return " ".join(words)

def create_vectorizer(train_texts, max_features=5000):
    """Create and fit TF-IDF vectorizer"""
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    X_train_vec = vectorizer.fit_transform(train_texts)
    # Save vectorizer
    with open("models/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    return vectorizer, X_train_vec

def load_vectorizer():
    """Load saved TF-IDF vectorizer"""
    with open("models/vectorizer.pkl", "rb") as f:
        return pickle.load(f)
