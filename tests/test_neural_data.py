"""Data-contract checks run without installing optional neural dependencies."""
import pandas as pd
import pytest

from src.neural_data import assert_partition_contract, masked_text


def test_neural_masking_removes_names_but_preserves_direction_and_context():
    text = masked_text('<e2>RareCauseName</e2> leads to <e1>SecretEffectName</e1>.')
    assert 'RareCauseName' not in text and 'SecretEffectName' not in text
    assert text.index('[E2]') < text.index('[E1]')
    assert 'leads to' in text
    with pytest.raises(ValueError):
        masked_text('<e1>incomplete</e1> has no second entity')


def test_partition_contract_rejects_leakage_and_changed_labels():
    frames = {k: pd.DataFrame({'group': [k], 'label': [i]})
              for i, k in enumerate(['train', 'dev', 'test'])}
    reference = pd.concat([v.assign(split=k) for k, v in frames.items()], ignore_index=True)
    assert_partition_contract(frames, reference)
    frames['dev'].loc[0, 'label'] = 2
    with pytest.raises(ValueError, match='differs'):
        assert_partition_contract(frames, reference)
    frames['dev'].loc[0, 'group'] = 'train'
    with pytest.raises(ValueError, match='leakage'):
        assert_partition_contract(frames, reference)
