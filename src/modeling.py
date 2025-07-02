from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

def build_model():
    pipeline = Pipeline([
        ("vectorizer", TfidfVectorizer()),
        ("classifier", LogisticRegression())
    ])
    return pipeline

if __name__ == "__main__":
    model = build_model()
    print(model)