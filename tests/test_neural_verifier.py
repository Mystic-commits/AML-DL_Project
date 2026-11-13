"""The portable verifier must detect tampering even if file hashes are regenerated."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.verify_neural_run import ROOT, verify
from src.extraction import relation_scores
from src.neural_data import file_sha


def fixture_run(path):
    reference = ROOT / 'artifacts/runs/phase1_extraction_verified_20261101/partitions.csv'
    frame = pd.read_csv(reference).query('split == "dev"')[['group','label']].reset_index(drop=True)
    frame['prediction'] = frame.label
    probs = np.eye(3)[frame.label.to_numpy()]
    for k in range(3): frame[f'p{k}'] = probs[:,k]
    frame.to_csv(path / 'dev_predictions.csv', index=False)
    score = relation_scores(frame.label, frame.prediction)
    history = [{'epoch':1, 'dev':score}, {'epoch':2, 'dev':score}]
    (path/'history.json').write_text(json.dumps(history))
    (path/'run.json').write_text(json.dumps({'variant':'fixture', 'history':history,
        'test_examples_evaluated':0, 'selected_epoch':1,
        'reference_partition_sha256':file_sha(reference),
        'selected_development_metrics':score, 'selected_checkpoint_reload_max_abs_error':0}))
    rehash(path)


def rehash(path):
    (path/'manifest.json').write_text(json.dumps({p.name:file_sha(p)
        for p in path.iterdir() if p.name != 'manifest.json'}))


def test_verifier_rejects_wrong_row_alignment_after_rehash(tmp_path):
    fixture_run(tmp_path)
    assert verify(tmp_path, portable=True)['status'] == 'PASS'
    frame = pd.read_csv(tmp_path/'dev_predictions.csv')
    frame.loc[0,'group'] = 'not-in-development'
    frame.to_csv(tmp_path/'dev_predictions.csv',index=False)
    rehash(tmp_path)
    with pytest.raises(ValueError, match='row order'):
        verify(tmp_path, portable=True)


def test_verifier_rejects_false_metric_claim_after_rehash(tmp_path):
    fixture_run(tmp_path)
    report = json.loads((tmp_path/'run.json').read_text())
    report['selected_development_metrics']['edge_f1'] = .2
    (tmp_path/'run.json').write_text(json.dumps(report))
    rehash(tmp_path)
    with pytest.raises(ValueError, match='metric mismatch'):
        verify(tmp_path, portable=True)
