

# model_1.py

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV

class ComplaintClassifier:
    def __init__(self, model_name="Logistic Regression"):
        self.model_name = model_name
        
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "Linear SVM": LinearSVC(class_weight="balanced"),
            "Naive Bayes": MultinomialNB(),
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
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not supported.")
        
        self.model = self.models[model_name]
        self.grid = None
        self.best_model = None
    
    def train(self, X_train, y_train):
        print(f"Training {self.model_name}...")
        self.model.fit(X_train, y_train)
        self.best_model = self.model
    
    def predict(self, X):
        if self.best_model is None:
            raise RuntimeError("Model not trained yet!")
        return self.best_model.predict(X)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        print(f"\n{self.model_name} Results:")
        print(f"Accuracy: {acc:.3f} | F1-macro: {f1:.3f}")
        print(classification_report(y_test, y_pred, zero_division=0))
        return {"Accuracy": acc, "F1-macro": f1}
    
    def grid_search(self, X_train, y_train, cv=3, scoring="f1_macro", n_jobs=-1, verbose=1):
        print(f"Running GridSearchCV for {self.model_name}...")
        self.grid = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grids[self.model_name],
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose
        )
        self.grid.fit(X_train, y_train)
        self.best_model = self.grid.best_estimator_
        print(f"\nBest Params: {self.grid.best_params_}")
        print(f"Best CV F1-macro: {self.grid.best_score_:.3f}")
        return self.grid.best_params_, self.grid.best_score_
