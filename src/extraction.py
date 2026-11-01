"""Classical directed relation classification on explicitly marked entity pairs.

Labels: 0=no causal relation, 1=e1 causes e2, 2=e2 causes e1.
Entity spans are GIVEN by SemEval, so this is not an end-to-end NER benchmark.
"""
import hashlib
import re

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold


def sentence_group(text):
    plain = re.sub(r'</?e[12]>', '', str(text))
    return hashlib.sha256(' '.join(plain.lower().split()).encode()).hexdigest()


def prepare(frame):
    required = {'sentence', 'relation'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Missing columns: {required - set(frame.columns)}')
    df = frame.copy()
    if df[list(required)].isna().any().any():
        raise ValueError('Missing sentence or relation')
    if not df.relation.isin(range(19)).all():
        raise ValueError('Relation must be a SemEval class ID from 0 to 18')
    valid = df.sentence.map(lambda x: all(len(re.findall(p, x)) == 1 for p in
        [r'<e1>.+?</e1>', r'<e2>.+?</e2>']))
    if not valid.all():
        raise ValueError('Exactly one nonempty e1 and e2 span required')
    df['label'] = df.relation.map(lambda x: 1 if x == 0 else (2 if x == 1 else 0))
    df['group'] = df.sentence.map(sentence_group)
    df['pair_key'] = df.sentence.map(lambda x: ' '.join(x.lower().split()))
    if (df.groupby('pair_key').label.nunique() > 1).any():
        raise ValueError('Conflicting labels for duplicate sentences')
    return df.drop_duplicates('pair_key').reset_index(drop=True)


def split_data(train, test):
    train, test = prepare(train), prepare(test)
    # Original corpus has no document IDs. Sentence grouping is the strongest available contract.
    test = test[~test.group.isin(train.group)].reset_index(drop=True)
    fit, dev = next(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(
        train, train.label, train.group))
    return {'train': train.iloc[fit].reset_index(drop=True),
            'dev': train.iloc[dev].reset_index(drop=True), 'test': test}


def features(text, context_only=False):
    """Replace entity strings to reduce memorization while preserving direction markers."""
    for n in [1, 2]:
        text = re.sub(fr'<e{n}>.*?</e{n}>', f' ENTITY{n} ', text)
    return text


def lexical_label(sentence):
    # Conservative baseline with given entity spans and explicit direction cues.
    e1, e2 = re.search(r'<e1>.*?</e1>', sentence), re.search(r'<e2>.*?</e2>', sentence)
    left, right = sorted([e1, e2], key=lambda x: x.start())
    between = sentence[left.end():right.start()].lower()
    if re.search(r'\b(not|may|might|could|if)\b', sentence, re.I):
        return 0
    forward = re.search(r'\b(causes?|caused|leads? to|led to|results? in|resulted in|produces?|produced|triggers?|triggered)\b', between)
    reverse = re.search(r'\b(caused by|due to|result of|results? from|induced by)\b', between)
    if reverse:
        return 2 if left is e1 else 1
    if forward:
        return 1 if left is e1 else 2
    return 0


def relation_scores(gold, predictions):
    gold, predictions = np.asarray(gold), np.asarray(predictions)
    tp = int(((gold == predictions) & (gold != 0)).sum())
    fp = int(((predictions != 0) & (gold != predictions)).sum())
    fn = int(((gold != 0) & (gold != predictions)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {'edge_precision': p, 'edge_recall': r,
            'edge_f1': 2*p*r/(p+r) if p+r else 0.0,
            'macro_f1': float(f1_score(gold, predictions, labels=[0, 1, 2], average='macro', zero_division=0)),
            'tp': tp, 'fp': fp, 'fn': fn,
            'confusion_matrix': confusion_matrix(gold, predictions, labels=[0, 1, 2]).tolist()}


def predicted_edge(sentence, label):
    if int(label) == 0:
        return None
    from src.causal import canonical
    entities = [canonical(re.search(fr'<e{n}>(.*?)</e{n}>', sentence)[1]) for n in [1, 2]]
    return tuple(entities if int(label) == 1 else entities[::-1])
