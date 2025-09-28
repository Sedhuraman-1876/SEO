import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from ml.preprocessing import preprocess
import joblib
import os

class RankingPredictor:
    def __init__(self):
        self.model = None
        self.tfidf = TfidfVectorizer(ngram_range=(1,2), max_df=0.9, min_df=1)

    def train(self, df, text_col='text', target_col='rank_score'):
        df[text_col] = df[text_col].astype(str).apply(preprocess)
        X = df[text_col].tolist()
        y = df[target_col].astype(float).tolist()
        pipe = make_pipeline(self.tfidf, LinearRegression())
        pipe.fit(X, y)
        self.model = pipe
        return self

    def predict(self, texts):
        if not isinstance(texts, list):
            texts = [texts]
            
        texts = [preprocess(t) for t in texts]
        return self.model.predict(texts)

    def save_model(self, filepath):
        """Save the trained model to a file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath):
        """Load a trained model from a file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        model = cls()
        model.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model