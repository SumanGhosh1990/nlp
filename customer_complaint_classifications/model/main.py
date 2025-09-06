# main.py
import os
os.chdir('/Users/sagnikgupta/Desktop/Python_Project/NLP Projects') ##--- Provide your path here
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from processing import load_data, preprocess_texts, vectorize_texts
from model_1 import ComplaintClassifier

sns.set(style="whitegrid")
RND = 42



DATA_PATH = r'/Users/sagnikgupta/Desktop/Python_Project/NLP Projects/rows.csv' ##--- Provide your path here

def main():
    # Load data
    df = load_data(DATA_PATH)
    print("Dataset shape:", df.shape)

    # Clean texts
    df['clean_text'] = preprocess_texts(df['Consumer complaint narrative'])

    # Vectorize
    X, vectorizer = vectorize_texts(df['clean_text'])
    y = df['Product']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RND, stratify=y
    )

    # Initialize model
    clf = ComplaintClassifier(model_name="Logistic Regression")

    # Train and evaluate baseline
    clf.train(X_train, y_train)
    baseline_results = clf.evaluate(X_test, y_test)

    # Hyperparameter tuning
    best_params, best_score = clf.grid_search(X_train, y_train)

    # Evaluate best model
    tuned_results = clf.evaluate(X_test, y_test)

    # Print results
    print("\n=== Summary ===")
    print(f"Baseline results: {baseline_results}")
    print(f"Tuned best params: {best_params}")
    print(f"Tuned results: {tuned_results}")

if __name__ == "__main__":
    main()
