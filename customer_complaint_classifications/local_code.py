import os
import re
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

sns.set(style="whitegrid")


class TextClassifier:
    def __init__(self, csv_path, random_state=42):
        self.csv_path = csv_path
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "Linear SVM": LinearSVC(class_weight="balanced"),
            "Naive Bayes": MultinomialNB()
        }
        self.param_grids = {
            "Logistic Regression": {
                "C": [0.1, 1, 10],
                "penalty": ["l2"],
                "solver": ["lbfgs"]
            },
            "Linear SVM": {
                "C": [0.1, 1, 10]
            },
            "Naive Bayes": {
                "alpha": [0.1, 0.5, 1.0]
            }
        }
        self.df = None
        self.X = None
        self.y = None
        self.X_train = self.X_test = self.y_train = self.y_test = None

    def load_and_prepare_data(self):
        print("[INFO] Loading data...")
        data = pd.read_csv(self.csv_path, low_memory=False)
        self.df = data[['Product', 'Consumer complaint narrative']].dropna().reset_index(drop=True)
        print("[INFO] Data loaded. Shape:", self.df.shape)

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = text.split()
        tokens = [t for t in tokens if t not in stop_words]
        return " ".join(tokens)

    def preprocess(self):
        print("[INFO] Preprocessing text...")
        self.df['clean_text'] = self.df['Consumer complaint narrative'].apply(self.clean_text)

    def vectorize(self):
        print("[INFO] Vectorizing text...")
        self.X = self.vectorizer.fit_transform(self.df['clean_text'])
        self.y = self.df['Product']

    def split_data(self):
        print("[INFO] Splitting data...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=self.random_state, stratify=self.y
        )

    def train_and_evaluate_basic(self):
        print("[INFO] Training basic models...")
        results = {}
        for name, model in self.models.items():
            print("="*60)
            print(f"Training {name}...")
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            acc = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average="macro")
            results[name] = {"Accuracy": acc, "F1-macro": f1}
            print(f"{name} - Accuracy: {acc:.3f}, F1-macro: {f1:.3f}")
            print(classification_report(self.y_test, y_pred, zero_division=0))
        return pd.DataFrame(results).T

    def train_with_grid_search(self):
        print("[INFO] Running GridSearchCV on models...")
        final_results = {}
        for name, model in self.models.items():
            print("="*60)
            print(f"Running GridSearchCV for {name}...")
            grid = GridSearchCV(
                estimator=model,
                param_grid=self.param_grids[name],
                cv=3,
                scoring="f1_macro",
                n_jobs=-1,
                verbose=1
            )
            grid.fit(self.X_train, self.y_train)
            best_model = grid.best_estimator_
            y_pred = best_model.predict(self.X_test)
            final_results[name] = {
                "Best Params": grid.best_params_,
                "CV F1-macro": grid.best_score_,
                "Test Accuracy": accuracy_score(self.y_test, y_pred),
                "Test F1-macro": f1_score(self.y_test, y_pred, average="macro")
            }
            print(f"\n{name} Best Params: {grid.best_params_}")
            print(f"Best CV F1-macro: {grid.best_score_:.3f}")
            print(f"Test Accuracy: {accuracy_score(self.y_test, y_pred):.3f}")
            print(f"Test F1-macro: {f1_score(self.y_test, y_pred, average='macro'):.3f}")
            print(classification_report(self.y_test, y_pred, zero_division=0))
        return pd.DataFrame(final_results).T

    def plot_results(self, results_df):
        print("[INFO] Plotting results...")
        results_df[["CV F1-macro", "Test Accuracy", "Test F1-macro"]].plot(
            kind="bar", figsize=(10, 6), legend=True
        )
        plt.title("Model Performance Comparison")
        plt.ylabel("Score")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.show()


# ======================================
# Main Execution Script
# ======================================

if __name__ == "__main__":
    csv_path = "/Users/sagnikgupta/Desktop/Python_Project/NLP Projects/rows.csv"

    classifier = TextClassifier(csv_path)
    classifier.load_and_prepare_data()
    classifier.preprocess()
    classifier.vectorize()
    classifier.split_data()

    print("\n===== Basic Model Performance =====")
    basic_results = classifier.train_and_evaluate_basic()
    print(basic_results)

    print("\n===== Grid Search Model Performance =====")
    tuned_results = classifier.train_with_grid_search()
    print(tuned_results)

    classifier.plot_results(tuned_results)