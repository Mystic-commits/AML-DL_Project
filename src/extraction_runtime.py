"""Portable multiclass TF-IDF linear inference without pickle loading."""
import json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def export_model(model, path):
    vec, clf = model['tfidf'], model['classifier']
    payload = {'format':1, 'classes':clf.classes_.tolist(),
        'vocabulary': {k:int(v) for k,v in vec.vocabulary_.items()}, 'idf':vec.idf_.tolist(),
        'token_pattern':vec.token_pattern, 'ngram_range':list(vec.ngram_range),
        'sublinear_tf':vec.sublinear_tf, 'coef':clf.coef_.tolist(), 'intercept':clf.intercept_.tolist()}
    Path(path).write_text(json.dumps(payload, separators=(',',':')))


class FrozenExtractor:
    def __init__(self, path):
        p = json.loads(Path(path).read_text())
        if p['format'] != 1 or p['classes'] != [0,1,2]:
            raise ValueError('Unsupported extraction model')
        self.vectorizer = TfidfVectorizer(vocabulary=p['vocabulary'], token_pattern=p['token_pattern'],
            ngram_range=tuple(p['ngram_range']), sublinear_tf=p['sublinear_tf'])
        self.vectorizer.idf_ = np.array(p['idf'])
        self.coef, self.intercept = np.array(p['coef']), np.array(p['intercept'])

    def predict(self, texts):
        return np.argmax(self.vectorizer.transform(texts) @ self.coef.T + self.intercept, axis=1)
