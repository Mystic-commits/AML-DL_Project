"""Recompute selected development metrics and check neural-run artifact integrity."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.extraction import relation_scores
from src.neural_data import file_sha

ROOT = Path(__file__).resolve().parents[1]


def verify(output, portable=False):
    output = Path(output)
    manifest = json.loads((output / 'manifest.json').read_text())
    for name, digest in manifest.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid manifest path')
        path = output / relative
        if portable and relative.parts[0] == 'checkpoint' and not path.exists():
            continue
        if not path.is_file() or file_sha(path) != digest:
            raise ValueError(f'Run hash mismatch: {name}')
    report = json.loads((output / 'run.json').read_text())
    history = json.loads((output / 'history.json').read_text())
    if report['history'] != history:
        raise ValueError('History mismatch')
    if report['test_examples_evaluated'] != 0:
        raise ValueError('Development training record must not contain test evaluation')
    best = max(history, key=lambda r: (r['dev']['edge_f1'], -r['epoch']))
    if report['selected_epoch'] != best['epoch']:
        raise ValueError('Development-only checkpoint selection mismatch')
    reference = ROOT / 'artifacts/runs/phase1_extraction_verified_20261101/partitions.csv'
    if file_sha(reference) != report['reference_partition_sha256']:
        raise ValueError('Frozen reference hash mismatch')
    reference = pd.read_csv(reference).query('split == "dev"').reset_index(drop=True)
    predictions = pd.read_csv(output / 'dev_predictions.csv')
    if not predictions[['group','label']].equals(reference[['group','label']]):
        raise ValueError('Development row order or labels differ from reference')
    probs = predictions[['p0','p1','p2']].to_numpy()
    if (not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any()
            or not np.allclose(probs.sum(axis=1), 1, atol=1e-5)):
        raise ValueError('Invalid saved probabilities')
    if not np.array_equal(predictions.prediction, probs.argmax(axis=1)):
        raise ValueError('Predictions disagree with saved probabilities')
    measured = relation_scores(predictions.label, predictions.prediction)
    for key, value in measured.items():
        if not np.allclose(value, report['selected_development_metrics'][key], atol=1e-12):
            raise ValueError(f'Saved development metric mismatch: {key}')
        if not np.allclose(value, best['dev'][key], atol=1e-12):
            raise ValueError(f'Best-epoch metric mismatch: {key}')
    if report['selected_checkpoint_reload_max_abs_error'] > 1e-5:
        raise ValueError('Checkpoint reload exceeded tolerance')
    return {'status': 'PASS', 'mode': 'portable_development_audit' if portable else 'full_development_audit',
            'variant': report['variant'], 'selected_epoch': best['epoch'],
            'development_edge_f1': measured['edge_f1'], 'test_examples_evaluated': 0,
            'checks': ['available hashes', 'development split alignment', 'selected epoch',
                       'saved probabilities', 'recomputed directed metrics', 'checkpoint reload receipt']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--portable', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(args.run, args.portable), indent=2))
