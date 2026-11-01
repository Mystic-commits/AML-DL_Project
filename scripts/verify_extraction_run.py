"""Independently reconstruct metrics, hashes, model selection and split isolation."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from src.extraction import features, relation_scores
from scripts.train_extraction import sha


def verify(out, portable=False):
    hashes = json.loads((out/'manifest.json').read_text())
    for name, digest in hashes.items():
        if not (out/name).exists() and portable and name in ['train.csv','dev.csv','test.csv','model.joblib']:
            continue
        if not (out/name).exists():
            raise FileNotFoundError(f'Regenerate local run asset: {name}')
        if sha(out/name) != digest:
            raise ValueError(f'Hash mismatch: {name}')
    if portable:
        membership = pd.read_csv(out/'partitions.csv')
        frames = {k: membership[membership.split == k].reset_index(drop=True) for k in ['train','dev','test']}
    else:
        frames = {k: pd.read_csv(out/f'{k}.csv') for k in ['train','dev','test']}
    for a,b in [('train','dev'),('train','test'),('dev','test')]:
        if set(frames[a].group) & set(frames[b].group):
            raise ValueError(f'Split leakage: {a}/{b}')
    report = json.loads((out/'metrics.json').read_text())
    selected = max(report['search'], key=lambda x:(x['dev']['edge_f1'],-x['C']))
    if selected['C'] != report['selected_C']:
        raise ValueError('Selection mismatch')
    predictions = pd.read_csv(out/'predictions.csv')
    test = frames['test']
    if not predictions.group.equals(test.group) or not predictions.label.equals(test.label):
        raise ValueError('Prediction alignment mismatch')
    if not portable:
        model = joblib.load(out/'model.joblib')
        if (out/'model_snapshot.json').exists():
            from src.extraction_runtime import FrozenExtractor
            if not np.array_equal(FrozenExtractor(out/'model_snapshot.json').predict(test.sentence.map(features)), predictions.tfidf_lr):
                raise ValueError('Portable prediction mismatch')
        if not np.array_equal(model.predict(test.sentence.map(features)), predictions.tfidf_lr):
            raise ValueError('Model prediction mismatch')
    for name, expected in report['test'].items():
        if relation_scores(predictions.label, predictions[name]) != expected:
            raise ValueError(f'Metric mismatch: {name}')
    result = {'status':'PASS','mode':'portable numeric audit' if portable else 'full model reproduction','checks':['available artifact hashes' if portable else 'all artifact hashes','split isolation','dev-only selection',
        *(['saved-model predictions'] if not portable else []),'all test metrics','prediction alignment']}
    (out/('verification_portable.json' if portable else 'verification.json')).write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--portable', action='store_true')
    args = parser.parse_args()
    verify(args.run, args.portable)
