import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
import joblib


def load_chat():
    return pd.read_csv("chat.csv")


def train_vectorizer(data):
    vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True, stop_words="english")
    vectorizer.fit_transform(data["message"])

    return vectorizer


def save_vectorizer(vectorizer):
    joblib.dump(vectorizer, "vectorizer.pkl")


def load_vectorizer() -> TfidfVectorizer:
    return joblib.load("vectorizer.pkl")


def train_hash_vectorizer(data):
    vectorizer = HashingVectorizer(stop_words="english")
    vectorizer.fit_transform(data["message"])

    return vectorizer


def extract_n_words(vectorizer, msg, n):
    tfidf_matrix = vectorizer.transform([msg])
    feature_names = vectorizer.get_feature_names_out()

    tfidf_scores = tfidf_matrix.toarray()[0]

    # Get the indices of the top N scores
    top_indices = tfidf_scores.argsort()[-n:][::-1]

    # Get the top N keywords and their scores
    top_keywords = [(feature_names[i], tfidf_scores[i]) for i in top_indices]

    return top_keywords


def word_cloud(vectorizer):
    pass
