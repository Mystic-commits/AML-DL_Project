"""Offline checks for a genuine frozen encoder and development checkpoint selection."""
import pytest
pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')
import torch
from src.neural import train_step
from src.neural_training import choose_best_epoch, configure_variant


def test_selection_uses_best_development_epoch_and_earliest_tie():
    rows = [{'epoch': 1, 'dev': {'edge_f1': .7}},
            {'epoch': 2, 'dev': {'edge_f1': .8}},
            {'epoch': 3, 'dev': {'edge_f1': .8}},
            {'epoch': 4, 'dev': {'edge_f1': .6}}]
    assert choose_best_epoch(rows) == 2


def test_frozen_encoder_stays_unchanged_while_head_learns():
    model = transformers.DistilBertForSequenceClassification(transformers.DistilBertConfig(
        vocab_size=12, dim=16, hidden_dim=24, n_layers=1, n_heads=2,
        num_labels=3, max_position_embeddings=16))
    details = configure_variant(model, 'frozen_encoder_weighted')
    assert 0 < details['trainable_parameters'] < details['total_parameters']
    before_base = {k: v.clone() for k, v in model.base_model.state_dict().items()}
    before_head = model.classifier.weight.detach().clone()
    batch = {'input_ids': torch.tensor([[2,3,4], [5,6,7]]),
             'attention_mask': torch.ones(2,3,dtype=torch.long), 'labels': torch.tensor([1,2])}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    train_step(model, batch, optimizer)
    assert not model.base_model.training
    assert all(torch.equal(before_base[k], v) for k, v in model.base_model.state_dict().items())
    assert not torch.equal(before_head, model.classifier.weight)
