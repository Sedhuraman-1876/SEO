import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from ml.preprocessing import preprocess
import joblib
import os

class IntentModels:
    def __init__(self):
        self.tfidf = TfidfVectorizer(ngram_range=(1,2), max_df=0.9, min_df=1)
        self.nb = MultinomialNB()
        self.dt = DecisionTreeClassifier(random_state=42)
        self.nb_pipe = None
        self.dt_pipe = None

    def train(self, df: pd.DataFrame, text_col='text', label_col='intent'):
        df[text_col] = df[text_col].astype(str).apply(preprocess)
        X = df[text_col].values
        y = df[label_col].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.nb_pipe = make_pipeline(self.tfidf, self.nb)
        self.dt_pipe = make_pipeline(self.tfidf, self.dt)

        self.nb_pipe.fit(X_train, y_train)
        self.dt_pipe.fit(X_train, y_train)

        y_pred_nb = self.nb_pipe.predict(X_test)
        y_pred_dt = self.dt_pipe.predict(X_test)
        
        print("\nNaive Bayes classification report:")
        print(classification_report(y_test, y_pred_nb))
        
        print("\nDecision Tree classification report:")
        print(classification_report(y_test, y_pred_dt))
        
        return self

    def predict(self, texts):
        if not isinstance(texts, list):
            texts = [texts]
            
        texts = [preprocess(t) for t in texts]
        return {
            "nb": self.nb_pipe.predict(texts).tolist(),
            "dt": self.dt_pipe.predict(texts).tolist()
        }

    def save_model(self, filepath):
        """Save the trained model to a file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'nb_pipe': self.nb_pipe,
            'dt_pipe': self.dt_pipe
        }, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath):
        """Load a trained model from a file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        data = joblib.load(filepath)
        model = cls()
        model.nb_pipe = data['nb_pipe']
        model.dt_pipe = data['dt_pipe']
        print(f"Model loaded from {filepath}")
        return model