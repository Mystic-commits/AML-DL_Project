"""Reproducible Phase 1 extraction experiment. Test labels never select a model."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.extraction import features, lexical_label, relation_scores, split_data

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    out = args.output or ROOT / 'artifacts/runs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_extraction')
    out.mkdir(parents=True, exist_ok=False)
    raw = ROOT / 'data/raw/semeval'
    frames = split_data(pd.read_parquet(raw / 'train.parquet'), pd.read_parquet(raw / 'test.parquet'))
    (out / 'data_manifest.json').write_bytes((raw / 'manifest.json').read_bytes())
    protocol = {'seed': 42, 'task': 'given-entity directed causal relation classification',
        'selection': 'max dev edge F1, ties prefer smaller C', 'C_grid': [0.1, 1.0, 10.0],
        'no_threshold_tuning': True, 'group': 'normalized sentence without entity tags',
        'limitations': ['No document identifiers supplied by corpus', 'General semantic relations, not a scientific/news corpus',
            'Given entity spans; NER and candidate recall not measured', 'Not the official 19-class SemEval metric']}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    for name, df in frames.items():
        df.to_csv(out / f'{name}.csv', index=False)
    train, dev, test = (frames[k] for k in ['train', 'dev', 'test'])
    x = {k: df.sentence.map(features) for k, df in frames.items()}
    traces, models = [], []
    for c in protocol['C_grid']:
        model = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True,
                    max_features=30000, token_pattern=r'(?u)\b\w+\b')),
            ('classifier', LogisticRegression(C=c, class_weight='balanced', max_iter=1500, random_state=42))])
        model.fit(x['train'], train.label)
        traces.append({'C': c, 'dev': relation_scores(dev.label, model.predict(x['dev']))})
        models.append(model)
    best = max(range(len(models)), key=lambda i: (traces[i]['dev']['edge_f1'], -traces[i]['C']))
    model = models[best]
    joblib.dump(model, out / 'model.joblib')
    from src.extraction_runtime import export_model
    export_model(model, out / 'model_snapshot.json')
    predictions = {'majority': np.zeros(len(test), dtype=int),
        'lexical': np.array([lexical_label(t) for t in test.sentence]),
        'tfidf_lr': model.predict(x['test'])}
    # Meaningful ablation: remove surrounding context and use only marked entity strings.
    entity = lambda t: ' '.join(__import__('re').findall(r'<e[12]>(.*?)</e[12]>', t))
    ablation = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2))),
        ('classifier', LogisticRegression(C=traces[best]['C'], class_weight='balanced', max_iter=1500, random_state=42))])
    ablation.fit(train.sentence.map(entity), train.label)
    predictions['entity_only_ablation'] = ablation.predict(test.sentence.map(entity))
    results = {name: relation_scores(test.label, pred) for name, pred in predictions.items()}
    saved = test[['group', 'label']].copy()
    for name, pred in predictions.items():
        saved[name] = pred
    saved.to_csv(out / 'predictions.csv', index=False)
    rng = np.random.default_rng(42)
    intervals = []
    for _ in range(1000):
        groups = test.group.unique()
        sampled = rng.choice(groups, len(groups), replace=True)
        idx = np.concatenate([np.flatnonzero(test.group.to_numpy() == g) for g in sampled])
        intervals.append(relation_scores(test.label.to_numpy()[idx], predictions['tfidf_lr'][idx])['edge_f1'])
    report = {'selected_C': traces[best]['C'], 'search': traces, 'test': results,
        'counts': {k: {'rows': len(v), 'labels': {str(c): int(n) for c,n in v.label.value_counts().items()}} for k,v in frames.items()},
        'test_edge_f1_bootstrap_95ci': np.quantile(intervals, [.025,.975]).tolist(),
        'ci_scope': 'Sentence-group bootstrap conditional on frozen model; document clustering unavailable'}
    (out / 'metrics.json').write_text(json.dumps(report, indent=2))
    environment = {'python': sys.version, 'platform': platform.platform(),
        'packages': {p: importlib.metadata.version(p) for p in ['pandas','scikit-learn','numpy','pgmpy']}}
    (out / 'environment.json').write_text(json.dumps(environment, indent=2))
    (out / 'requirements-freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'], text=True))
    code = out / 'code'
    code.mkdir()
    for path in [ROOT/'src/extraction.py', ROOT/'scripts/train_extraction.py']:
        (code/path.name).write_bytes(path.read_bytes())
    files = {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}
    (out/'manifest.json').write_text(json.dumps(files, indent=2))
    print(json.dumps(report, indent=2))
    print(f'RUN={out}')


if __name__ == '__main__':
    main()
