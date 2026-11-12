"""Offline tests use a tiny random transformer, never a downloaded model."""
import numpy as np
import pytest

# Optional requirements are isolated from the current hosted application.
torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')
from tokenizers import Tokenizer, models, pre_tokenizers
from src.neural import balanced_weights, probabilities, seed_everything, tokenise_checked, train_step


def tiny_tokenizer():
    vocab = {v: i for i, v in enumerate(['[PAD]', '[UNK]', 'word', 'causes'])}
    backend = Tokenizer(models.WordLevel(vocab, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]', unk_token='[UNK]')
    tokenizer.add_special_tokens({'additional_special_tokens': ['[E1]', '[E2]']})
    # DistilBERT does not accept BERT-style token_type_ids.
    tokenizer.model_input_names = ['input_ids', 'attention_mask']
    return tokenizer


def test_truncation_cannot_drop_either_entity():
    tok = tiny_tokenizer()
    tokenise_checked(tok, ['[E2] causes [E1]'], 8)
    with pytest.raises(ValueError, match='lost'):
        tokenise_checked(tok, ['[E1] ' + 'word ' * 20 + '[E2]'], 8)
    with pytest.raises(ValueError, match='exactly one'):
        tokenise_checked(tok, ['[E1] causes [E1]'], 8)


def test_weighted_backprop_and_safetensors_reload(tmp_path):
    seed_everything(42)
    tok = tiny_tokenizer()
    config = transformers.DistilBertConfig(vocab_size=len(tok), dim=24, hidden_dim=32,
        n_layers=1, n_heads=2, max_position_embeddings=32, num_labels=3,
        dropout=0, attention_dropout=0, seq_classif_dropout=0)
    model = transformers.DistilBertForSequenceClassification(config)
    encoded = tok.pad(tokenise_checked(tok, ['[E1] causes [E2]', '[E2] causes [E1]'], 16), return_tensors='pt')
    batch = dict(encoded, labels=torch.tensor([1, 2]))
    before = model.classifier.weight.detach().clone()
    weights = balanced_weights([0, 0, 0, 1, 2])
    assert weights[1] > weights[0]
    loss = train_step(model, batch, torch.optim.AdamW(model.parameters(), lr=1e-3), weights)
    assert np.isfinite(loss)
    assert not torch.equal(before, model.classifier.weight)
    expected = probabilities(model, encoded, 'cpu')
    assert np.allclose(expected.sum(axis=1), 1)
    model.save_pretrained(tmp_path, safe_serialization=True)
    restored = transformers.DistilBertForSequenceClassification.from_pretrained(tmp_path, use_safetensors=True)
    assert np.allclose(expected, probabilities(restored, encoded, 'cpu'), atol=1e-6)
