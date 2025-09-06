# processing.py

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def load_data(path):
    df = pd.read_csv(path, low_memory=False)
    df = df[['Product', 'Consumer complaint narrative']].dropna().reset_index(drop=True)
    return df

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # keep only letters and spaces
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    return " ".join(tokens)

def preprocess_texts(texts):
    return texts.apply(clean_text)

def vectorize_texts(texts, max_features=5000, ngram_range=(1,2)):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X = vectorizer.fit_transform(texts)
    return X, vectorizer
