"""Optional DistilBERT components; importing this module does not load model weights."""
import random

import numpy as np


def choose_device(requested='auto'):
    import torch
    if requested == 'auto':
        return 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    if requested not in {'cpu', 'cuda', 'mps'}:
        raise ValueError('Device must be auto, cpu, cuda or mps')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable')
    if requested == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('Apple MPS is unavailable')
    return requested


def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def tokenise_checked(tokenizer, texts, max_length):
    """Reject malformed/truncated markers rather than silently changing the task."""
    if max_length < 4:
        raise ValueError('Token budget must leave room for both entities')
    markers = ['[E1]', '[E2]']
    marker_ids = [tokenizer.convert_tokens_to_ids(m) for m in markers]
    if len(set(marker_ids)) != 2 or tokenizer.unk_token_id in marker_ids:
        raise ValueError('Register both entity markers before tokenisation')
    if any(any(text.count(m) != 1 for m in markers) for text in texts):
        raise ValueError('Each input needs exactly one E1 and one E2 marker')
    encoded = tokenizer(list(texts), truncation=True, max_length=max_length, padding=False)
    for index, ids in enumerate(encoded['input_ids']):
        if any(ids.count(marker_id) != 1 for marker_id in marker_ids):
            raise ValueError(f'Entity marker lost after tokenisation at row {index}; increase max_length')
    return encoded


def balanced_weights(labels):
    """Calculate weights from training labels only, without oversampling."""
    import torch
    labels = np.asarray(labels, dtype=int)
    if not len(labels) or not np.isin(labels, [0, 1, 2]).all():
        raise ValueError('Expected training labels 0, 1 and 2')
    counts = np.bincount(labels, minlength=3)
    if (counts == 0).any():
        raise ValueError('All three classes must appear in training data')
    return torch.tensor(len(labels) / (3 * counts), dtype=torch.float32)


def load_pretrained(config):
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    revision = config['model_revision']
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('Pin a full upstream model commit before loading')
    common = dict(revision=revision, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(config['model_id'], **common)
    tokenizer.add_special_tokens({'additional_special_tokens': config['entity_markers']})
    model_config = AutoConfig.from_pretrained(config['model_id'], **common)
    model_config.num_labels = 3
    model_config.id2label = {int(k): v for k, v in config['labels'].items()}
    model_config.label2id = {v: k for k, v in model_config.id2label.items()}
    model_config.dropout = config['dropout']
    model_config.attention_dropout = config['dropout']
    model_config.seq_classif_dropout = config['dropout']
    model = AutoModelForSequenceClassification.from_pretrained(
        config['model_id'], config=model_config, use_safetensors=True,
        attn_implementation='eager', **common)
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    return tokenizer, model


def train_step(model, batch, optimizer, weights=None, max_grad_norm=1.0):
    import torch
    from torch.nn import functional as F
    model.train()
    optimizer.zero_grad(set_to_none=True)
    labels = batch['labels']
    logits = model(**{k: v for k, v in batch.items() if k != 'labels'}).logits
    loss = F.cross_entropy(logits, labels, weight=weights)
    if not torch.isfinite(loss):
        raise ValueError('Non-finite training loss')
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach().cpu())


def probabilities(model, encoded, device):
    import torch
    model.eval()
    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in encoded.items() if k != 'labels'}
        return model(**inputs).logits.softmax(-1).cpu().numpy()
