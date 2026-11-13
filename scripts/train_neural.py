"""Fine-tune or freeze DistilBERT; this command cannot evaluate the test split."""
import argparse
import gc
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.neural import choose_device, load_pretrained, seed_everything
from src.neural_data import file_sha
from src.neural_training import VARIANTS, evaluate, make_loader, train_epochs

ROOT = Path(__file__).resolve().parents[1]


def main():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/phase2_extraction.json')
    parser.add_argument('--data', type=Path, default=ROOT / 'data/processed/phase2_extraction')
    parser.add_argument('--variant', choices=VARIANTS, default=VARIANTS[0])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new run directory; completed or partial runs are never overwritten')
    config = json.loads(args.config.read_text())
    if args.seed not in config['seeds']:
        raise ValueError('Seed must be declared in the experiment protocol')
    expected_audit = json.loads((ROOT / 'artifacts/phase2/data_audit.json').read_text())
    local_audit = json.loads((args.data / 'data_audit.json').read_text())
    if local_audit != expected_audit:
        raise ValueError('Prepared dataset does not match the committed data audit')
    frames = {}
    for split in ['train', 'dev']:
        path = args.data / f'{split}.csv'
        if file_sha(path) != expected_audit['local_files_sha256'][path.name]:
            raise ValueError(f'Prepared {split} hash mismatch')
        frames[split] = pd.read_csv(path)
    # No test.csv or test labels are opened by this runner.
    if set(frames['train'].group) & set(frames['dev'].group):
        raise ValueError('Training/development group leakage')
    seed_everything(args.seed)
    device = choose_device(args.device)
    print(f'RUN {args.variant} seed={args.seed} device={device}; train={len(frames["train"])} dev={len(frames["dev"])} test=UNTOUCHED', flush=True)
    tokenizer, model = load_pretrained(config)
    result, selected_probs = train_epochs(model, tokenizer, frames['train'], frames['dev'],
        config, args.variant, args.seed, args.output, device)
    model.to('cpu')
    del model
    gc.collect()
    if device == 'mps':
        torch.mps.empty_cache()
    checkpoint = args.output / 'checkpoint'
    restored_tok = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    restored = AutoModelForSequenceClassification.from_pretrained(checkpoint,
        local_files_only=True, trust_remote_code=False, use_safetensors=True).to(device)
    loader = make_loader(frames['dev'], restored_tok, config['max_length'], config['batch_size'], False, args.seed)
    metrics, restored_probs = evaluate(restored, loader, device)
    reload_error = float(np.abs(restored_probs - selected_probs).max())
    if not np.allclose(restored_probs, selected_probs, atol=1e-5):
        raise ValueError(f'Checkpoint roundtrip mismatch: {reload_error}')
    result.update({
        'config': config, 'training_device': device,
        'selected_development_metrics': metrics,
        'selected_checkpoint_reload_max_abs_error': reload_error,
        'source_hashes': {str(p.relative_to(ROOT)): file_sha(p) for p in [
            ROOT / 'src/neural.py', ROOT / 'src/neural_training.py',
            ROOT / 'scripts/train_neural.py', args.config]},
        'input_hashes': {f'{name}.csv': expected_audit['local_files_sha256'][f'{name}.csv'] for name in ['train','dev']},
        'reference_partition_sha256': expected_audit['frozen_reference_partition_sha256'],
        'git_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        'environment': {'python': sys.version, 'platform': platform.platform(),
            'packages': {p: importlib.metadata.version(p) for p in ['torch','transformers','numpy','pandas','scikit-learn']}},
        'limitations': ['One seed in this run', 'Development-selected performance is not a held-out test result',
                        'Given entity spans and sentence-level split groups'],
    })
    (args.output / 'run.json').write_text(json.dumps(result, indent=2))
    manifest = {str(p.relative_to(args.output)): file_sha(p) for p in args.output.rglob('*') if p.is_file()}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'status': result['status'], 'variant': args.variant,
        'selected_epoch': result['selected_epoch'], 'development': metrics,
        'reload_max_abs_error': reload_error, 'test_examples_evaluated': 0}, indent=2), flush=True)


if __name__ == '__main__':
    main()
