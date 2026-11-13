"""Epoch training and development-only selection for the optional neural extractor."""
import json
from pathlib import Path

import numpy as np

from src.extraction import relation_scores
from src.neural import balanced_weights, tokenise_checked, train_step

VARIANTS = ('fine_tuned_weighted', 'frozen_encoder_weighted', 'fine_tuned_unweighted')


def configure_variant(model, variant):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown variant: {variant}')
    frozen = variant == 'frozen_encoder_weighted'
    for parameter in model.base_model.parameters():
        parameter.requires_grad_(not frozen)
    return {'encoder_frozen': frozen, 'weighted_loss': variant != 'fine_tuned_unweighted',
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)}


def make_loader(frame, tokenizer, max_length, batch_size, shuffle, seed):
    import torch
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding
    encoded = tokenise_checked(tokenizer, frame.model_input.tolist(), max_length)
    rows = [{**{key: value[i] for key, value in encoded.items()}, 'labels': int(label)}
            for i, label in enumerate(frame.label)]
    return DataLoader(rows, batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed), num_workers=0,
                      collate_fn=DataCollatorWithPadding(tokenizer=tokenizer))


def evaluate(model, loader, device):
    import torch
    from torch.nn import functional as F
    model.eval()
    probabilities, gold = [], []
    loss_sum, rows = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop('labels').to(device)
            logits = model(**{key: value.to(device) for key, value in batch.items()}).logits
            loss_sum += float(F.cross_entropy(logits, labels, reduction='sum').cpu())
            rows += len(labels)
            probabilities.append(logits.softmax(-1).cpu().numpy())
            gold.extend(labels.cpu().tolist())
    if not rows:
        raise ValueError('Cannot evaluate an empty development split')
    probs = np.concatenate(probabilities)
    if not np.isfinite(probs).all() or not np.allclose(probs.sum(axis=1), 1, atol=1e-5):
        raise ValueError('Invalid model probabilities')
    return {'unweighted_cross_entropy': loss_sum / rows,
            **relation_scores(gold, probs.argmax(axis=1))}, probs


def choose_best_epoch(history):
    """Ties use the earliest epoch, never test labels or the final epoch by default."""
    if not history:
        raise ValueError('No completed development evaluations')
    return max(history, key=lambda row: (row['dev']['edge_f1'], -row['epoch']))['epoch']


def train_epochs(model, tokenizer, train, dev, config, variant, seed, output, device):
    import time
    import torch
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    details = configure_variant(model, variant)
    model.to(device)
    train_loader = make_loader(train, tokenizer, config['max_length'], config['batch_size'], True, seed)
    dev_loader = make_loader(dev, tokenizer, config['max_length'], config['batch_size'], False, seed)
    weights = balanced_weights(train.label).to(device) if details['weighted_loss'] else None
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=config['learning_rate'], weight_decay=config['weight_decay'])
    history, best_score, selected_probs = [], -1, None
    started = time.perf_counter()
    for epoch in range(1, config['epochs'] + 1):
        losses = []
        epoch_started = time.perf_counter()
        for step, batch in enumerate(train_loader, 1):
            batch = {key: value.to(device) for key, value in batch.items()}
            losses.append(train_step(model, batch, optimizer, weights, config['max_grad_norm']))
            if step == 1 or step % 50 == 0:
                print(f'{variant} epoch {epoch}/{config["epochs"]} batch {step}/{len(train_loader)} loss={np.mean(losses):.4f}', flush=True)
        metrics, probs = evaluate(model, dev_loader, device)
        row = {'epoch': epoch, 'mean_training_batch_loss': float(np.mean(losses)),
               'dev': metrics, 'epoch_seconds': round(time.perf_counter() - epoch_started, 2)}
        history.append(row)
        if metrics['edge_f1'] > best_score:
            best_score = metrics['edge_f1']
            selected_probs = probs.copy()
            model.save_pretrained(output / 'checkpoint', safe_serialization=True)
            tokenizer.save_pretrained(output / 'checkpoint')
            predictions = dev[['group', 'label']].copy()
            predictions['prediction'] = probs.argmax(axis=1)
            for label in range(3):
                predictions[f'p{label}'] = probs[:, label]
            predictions.to_csv(output / 'dev_predictions.csv', index=False)
        (output / 'history.json').write_text(json.dumps(history, indent=2))
        print(f'EPOCH COMPLETE {json.dumps(row)}', flush=True)
    result = {'status': 'TRAINED_DEVELOPMENT_ONLY', 'variant': variant, 'seed': seed,
              'selected_epoch': choose_best_epoch(history), 'history': history,
              'parameter_counts': details, 'training_class_weights': None if weights is None else weights.cpu().tolist(),
              'elapsed_seconds': round(time.perf_counter() - started, 2),
              'test_examples_evaluated': 0,
              'loss_note': 'Training is mean batch loss; development CE is unweighted per-example loss.',
              'frozen_encoder_note': 'Frozen encoder remains in eval mode; classification head is trained.'}
    return result, selected_probs
