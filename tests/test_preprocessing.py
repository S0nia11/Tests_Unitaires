import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocessing import preprocess

def test_preprocess_lowercase():
    text = "This is A TEST"
    processed = preprocess(text)
    assert processed == processed.lower()

def test_preprocess_removes_urls():
    text = "Visit https://example.com now!"
    processed = preprocess(text)
    assert "http" not in processed and "https" not in processed and "www" not in processed

def test_preprocess_removes_stopwords():
    text = "I am running easily"
    processed = preprocess(text)
    words = processed.split()
    assert "i" not in words
    assert "am" not in words

def test_preprocess_removes_punctuation_and_digits():
    text = "This is 100% good!!!"
    processed = preprocess(text)
    for char in "0123456789%!?":
        assert char not in processed
