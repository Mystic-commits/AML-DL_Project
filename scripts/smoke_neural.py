"""Engineering check of a real pretrained model; never reports benchmark quality."""
import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.neural import (balanced_weights, choose_device, load_pretrained,
                        probabilities, seed_everything, tokenise_checked, train_step)
from src.neural_data import file_sha

ROOT = Path(__file__).resolve().parents[1]


def main():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/phase2_extraction.json')
    parser.add_argument('--data', type=Path, default=ROOT / 'data/processed/phase2_extraction')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--steps', type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a fresh smoke-run output directory')
    if not 1 <= args.steps <= 8:
        raise ValueError('Engineering smoke run must use 1..8 steps, not full training')
    config = json.loads(args.config.read_text())
    audit = json.loads((args.data / 'data_audit.json').read_text())
    # Validate the training data against its frozen local data contract.
    train_path = args.data / 'train.csv'
    if file_sha(train_path) != audit['local_files_sha256']['train.csv']:
        raise ValueError('Prepared training input hash mismatch')
    train = pd.read_csv(train_path)
    seed = config['seeds'][0]
    seed_everything(seed)
    device = choose_device(args.device)
    print(f'Loading pinned {config["model_id"]} on {device}; no test evaluation.', flush=True)
    started = time.perf_counter()
    tokenizer, model = load_pretrained(config)
    # Only training text is used to check the token budget; no test tuning.
    raw_tokens = tokenizer(train.model_input.tolist(), truncation=False, padding=False)
    token_lengths = np.array([len(ids) for ids in raw_tokens['input_ids']])
    tokenise_checked(tokenizer, train.model_input.tolist(), config['max_length'])
    print(f'Training marker coverage PASS; longest training input: {int(token_lengths.max())} tokens.', flush=True)
    # Balanced sample just exercises every output class. It is not a performance estimate.
    samples = pd.concat([train[train.label == label].sample(8, random_state=seed)
                         for label in [0, 1, 2]], ignore_index=True)
    weights = balanced_weights(train.label).to(device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                  weight_decay=config['weight_decay'])
    before = model.classifier.weight.detach().cpu().clone()
    losses = []
    for step in range(args.steps):
        rows = samples.iloc[[(step * 2 + i) % 8 + cls * 8 for cls in [0, 1, 2] for i in [0, 1]]]
        encoded = tokenise_checked(tokenizer, rows.model_input.tolist(), config['max_length'])
        batch = dict(tokenizer.pad(encoded, padding=True, return_tensors='pt'))
        batch['labels'] = torch.tensor(rows.label.to_numpy(), dtype=torch.long)
        batch = {k: v.to(device) for k, v in batch.items()}
        losses.append(train_step(model, batch, optimizer, weights, config['max_grad_norm']))
        print(f'Smoke training step {step + 1}/{args.steps}: finite loss {losses[-1]:.6f}', flush=True)
    if torch.equal(before, model.classifier.weight.detach().cpu()):
        raise ValueError('Classification weights did not update')
    # Compare the saved and reloaded checkpoint on CPU to avoid cross-device tolerances.
    model.to('cpu')
    encoded = tokenizer.pad(tokenise_checked(tokenizer, samples.model_input.iloc[:3].tolist(),
                                             config['max_length']), return_tensors='pt')
    expected = probabilities(model, encoded, 'cpu')
    if not np.isfinite(expected).all() or not np.allclose(expected.sum(axis=1), 1):
        raise ValueError('Invalid output probability distribution')
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = args.output / 'checkpoint'
    model.save_pretrained(checkpoint, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint)
    restored_tok = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    restored = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=False, use_safetensors=True)
    restored_encoded = restored_tok.pad(tokenise_checked(restored_tok,
        samples.model_input.iloc[:3].tolist(), config['max_length']), return_tensors='pt')
    actual = probabilities(restored, restored_encoded, 'cpu')
    reload_error = float(np.abs(actual - expected).max())
    if not np.allclose(actual, expected, atol=1e-6):
        raise ValueError(f'Checkpoint roundtrip mismatch: {reload_error}')
    report = {
        'status': 'PASS', 'run_kind': 'engineering_smoke_only_not_a_trained_release',
        'benchmark_metrics': None, 'test_examples_evaluated': 0,
        'model_id': config['model_id'], 'model_revision': config['model_revision'],
        'config_sha256': file_sha(args.config), 'train_sha256': file_sha(train_path),
        'reference_partition_sha256': audit['frozen_reference_partition_sha256'],
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'seed': seed, 'training_device': device, 'steps': args.steps,
        'smoke_sample_size': len(samples), 'classification_weights_changed': True,
        'finite_step_losses': losses, 'all_training_entity_markers_preserved': True,
        'training_max_tokens': int(token_lengths.max()),
        'training_token_length_p95': float(np.quantile(token_lengths, .95)),
        'checkpoint_roundtrip_max_abs_error': reload_error,
        'probability_rows_sum_to_one': True,
        'elapsed_seconds': round(time.perf_counter() - started, 2),
        'python': sys.version, 'platform': platform.platform(),
        'packages': {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'safetensors', 'tokenizers']},
        'checkpoint_files_sha256': {p.name: file_sha(p) for p in checkpoint.iterdir() if p.is_file()},
        'limitations': ['Only a few optimizer steps', 'No development or test model-quality evaluation',
                        'Not eligible for hosted release or academic score claims'],
    }
    (args.output / 'smoke_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
