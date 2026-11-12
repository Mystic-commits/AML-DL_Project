"""Freeze neural inputs against the historical classical extraction partition."""
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.extraction import features, prepare, split_data


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def masked_text(sentence):
    """Match the classical entity masking, with two explicit transformer markers."""
    # Reuse schema checks so malformed/empty/repeated entity spans fail before inference.
    prepare(pd.DataFrame({'sentence': [sentence], 'relation': [0]}))
    return features(sentence).replace('ENTITY1', '[E1]').replace('ENTITY2', '[E2]')


def assert_partition_contract(frames, reference):
    """Require identical row order, groups and labels to the frozen ML run."""
    membership = pd.concat([
        frame[['group', 'label']].assign(split=name)
        for name, frame in frames.items()
    ], ignore_index=True)
    for left, right in [('train', 'dev'), ('train', 'test'), ('dev', 'test')]:
        if set(frames[left].group) & set(frames[right].group):
            raise ValueError(f'Split leakage: {left}/{right}')
    if not membership.reset_index(drop=True).equals(reference.reset_index(drop=True)):
        raise ValueError('Neural membership differs from frozen classical comparison')
    return membership


def prepare_data(raw, output, reference_run):
    raw, output, reference_run = map(Path, (raw, output, reference_run))
    if output.exists():
        raise FileExistsError(f'Choose a fresh prepared-data directory: {output}')
    source = json.loads((raw / 'manifest.json').read_text())
    original = {}
    checks = {}
    for name in ['train', 'test']:
        path = raw / f'{name}.parquet'
        if file_sha(path) != source['files'][name]['sha256']:
            raise ValueError(f'Source hash mismatch: {name}')
        original[name] = pd.read_parquet(path)
        frame = original[name]
        cleaned = prepare(frame)
        checks[name] = {
            'raw_rows': len(frame),
            'missing_sentence': int(frame.sentence.isna().sum()),
            'missing_relation': int(frame.relation.isna().sum()),
            'duplicate_pairs_removed': len(frame) - len(cleaned),
        }
    frames = split_data(original['train'], original['test'])
    reference = pd.read_csv(reference_run / 'partitions.csv')
    membership = assert_partition_contract(frames, reference)
    # All validation finishes before materialising a directory.
    for frame in frames.values():
        frame['model_input'] = frame.sentence.map(masked_text)
    output.mkdir(parents=True, exist_ok=False)
    for name, frame in frames.items():
        frame.to_csv(output / f'{name}.csv', index=False)
    membership.to_csv(output / 'partitions.csv', index=False)
    audit = {
        'schema_version': 1,
        'dataset': source['dataset'], 'dataset_revision': source['revision'],
        'source_files': source['files'], 'raw_checks': checks,
        'test_sentence_overlap_removed': len(prepare(original['test'])) - len(frames['test']),
        'frozen_reference_partition_sha256': file_sha(reference_run / 'partitions.csv'),
        'identical_classical_partition': True,
        'pairwise_group_overlap': {'train_dev': 0, 'train_test': 0, 'dev_test': 0},
        'counts': {name: {'rows': len(frame), 'labels': {
            str(k): int(v) for k, v in frame.label.value_counts().sort_index().items()
        }} for name, frame in frames.items()},
        'local_files_sha256': {p.name: file_sha(p) for p in sorted(output.glob('*.csv'))},
        'representation': 'Same entity-masked context as ML, with [E1] and [E2] special tokens',
        'scope': 'Data verification only. No training or test prediction results.',
        'limitations': ['Given entity spans', 'Sentence groups only; no document identifiers'],
    }
    (output / 'data_audit.json').write_text(json.dumps(audit, indent=2))
    return audit
