import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from modeling import build_model
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def test_pipeline_type():
    pipeline = build_model()
    assert isinstance(pipeline, Pipeline)

def test_pipeline_steps():
    pipeline = build_model()
    steps = dict(pipeline.named_steps)
    assert 'vectorizer' in steps
    assert 'classifier' in steps
    assert isinstance(steps['vectorizer'], TfidfVectorizer)
    assert isinstance(steps['classifier'], LogisticRegression)

def test_pipeline_fit_predict():
    pipeline = build_model()
    X = ["fire in the forest", "I love pizza", "earthquake in Japan", "just a sunny day"]
    y = [1, 0, 1, 0]
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert len(preds) == len(X)
