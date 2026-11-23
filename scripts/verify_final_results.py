"""Independent, portable reconstruction of all reported held-out edge metrics."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.causal import build_graph, oracle
from src.extraction import relation_scores
from src.neural_data import file_sha

ROOT=Path(__file__).resolve().parents[1]


def verify(folder):
    folder=Path(folder)
    manifest=json.loads((folder/'manifest.json').read_text())
    for name,digest in manifest.items():
        path=folder/name
        if Path(name).name!=name:raise ValueError('Invalid manifest path')
        if name.endswith('_probabilities.npy') and not path.exists():continue
        if not path.is_file() or file_sha(path)!=digest:raise ValueError(f'Hash mismatch: {name}')
    report=json.loads((folder/'metrics.json').read_text())
    pred=pd.read_csv(folder/'predictions.csv')
    reference=pd.read_csv(ROOT/'artifacts/runs/phase1_extraction_verified_20261101/partitions.csv').query('split == "test"').reset_index(drop=True)
    if not pred[['group','label']].equals(reference[['group','label']]):raise ValueError('Test alignment mismatch')
    for model,expected in report['test'].items():
        measured=relation_scores(pred.label,pred[model])
        for key,value in measured.items():
            if not np.allclose(value,expected[key],atol=1e-12):raise ValueError(f'Metric mismatch: {model}/{key}')
        prob=folder/f'{model}_probabilities.npy'
        if prob.exists():
            values=np.load(prob,allow_pickle=False)
            if not np.array_equal(values.argmax(axis=1),pred[model]):raise ValueError('Probability/prediction mismatch')
    if len(pred)!=report['test_rows']:raise ValueError('Test row count mismatch')
    seed_scores=[report['test'][f'fine_tuned_weighted_seed{s}']['edge_f1'] for s in [42,17,2026]]
    if not np.isclose(np.mean(seed_scores),report['seed_stability']['mean']):raise ValueError('Seed mean mismatch')
    if not np.isclose(np.std(seed_scores,ddof=1),report['seed_stability']['sample_sd']):raise ValueError('Seed SD mismatch')
    case=report['controlled_graph_case'];graph,accepted,rejected=build_graph(case['raw_candidates'])
    if [(x['cause'],x['effect']) for x in accepted] != [(x['cause'],x['effect']) for x in case['accepted']]:raise ValueError('Graph acceptance mismatch')
    if len(rejected)!=len(case['rejected']):raise ValueError('Graph rejection mismatch')
    for query in case['queries']:
        if not np.isclose(oracle(query['sprinkler']),query['oracle']):raise ValueError('Oracle mismatch')
    return {'status':'PASS','test_rows':len(pred),'models_reconstructed':len(report['test']),
        'checks':['available hashes','test alignment','all directed metrics','seed statistics','graph acceptance','known intervention oracles'],
        'scope':'Numeric artifact audit; full neural prediction reproduction requires the released model or local checkpoints'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path);args=parser.parse_args()
    print(json.dumps(verify(args.folder),indent=2))
