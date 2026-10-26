"""Portable inference snapshot for the selected classical text model."""
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer


def export_snapshot(pipeline, path):
    vectorizer = pipeline['tfidf']
    classifier = pipeline['model']
    if not hasattr(classifier, 'coef_'):
        raise ValueError('The application snapshot requires a linear binary classifier.')
    payload = {
        'format': 1,
        'model': 'TF-IDF + Logistic Regression',
        'token_pattern': vectorizer.token_pattern,
        'ngram_range': list(vectorizer.ngram_range),
        'sublinear_tf': vectorizer.sublinear_tf,
        'vocabulary': {term: int(index) for term, index in vectorizer.vocabulary_.items()},
        'idf': vectorizer.idf_.tolist(),
        'coefficients': classifier.coef_[0].tolist(),
        'intercept': float(classifier.intercept_[0]),
        'classes': classifier.classes_.tolist(),
    }
    Path(path).write_text(json.dumps(payload, separators=(',', ':')) + '\n')


class FrozenReasoner:
    def __init__(self, path):
        payload = json.loads(Path(path).read_text())
        if payload['format'] != 1 or payload['classes'] != [0, 1]:
            raise ValueError('Unsupported classifier snapshot.')
        self.vectorizer = TfidfVectorizer(
            vocabulary=payload['vocabulary'],
            token_pattern=payload['token_pattern'],
            ngram_range=tuple(payload['ngram_range']),
            sublinear_tf=payload['sublinear_tf'],
        )
        self.vectorizer.idf_ = np.asarray(payload['idf'])
        self.coefficients = np.asarray(payload['coefficients'])
        self.intercept = payload['intercept']

    def predict_proba(self, texts):
        features = self.vectorizer.transform(texts)
        positive = expit(features @ self.coefficients + self.intercept)
        return np.column_stack((1 - positive, positive))

    def predict(self, texts):
        return (self.predict_proba(texts)[:, 1] > .5).astype(int)
