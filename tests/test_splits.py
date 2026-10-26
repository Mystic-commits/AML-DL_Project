"""
Tests for dataset split integrity and premise-disjoint guarantees.
Project 1: Neuro-Symbolic Causal Inference (Phase 1)
Team T094, Section E
"""

from pathlib import Path
import pytest
from src.experiment import load_splits

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_EXISTS = all(
    (ROOT / f"data/raw/{s}.csv").exists() for s in ["train", "dev", "test"]
)


@pytest.mark.skipif(
    not RAW_DATA_EXISTS,
    reason="Raw dataset CSVs not found in data/raw/. Run 'python scripts/fetch_data.py' to download.",
)
def test_no_exact_premise_overlap():
    """
    Verify that our premise-based grouping eliminates verbatim premise leakage
    across train, dev, and test sets, and preserves binary class balance.
    """
    splits, _ = load_splits()
    groups = [set(splits[s]["group"]) for s in ["train", "dev", "test"]]

    # Ensure no pairwise overlap of premise groups between splits
    assert not (groups[0] & groups[1]), "Leakage detected between train and dev!"
    assert not (groups[0] & groups[2]), "Leakage detected between train and test!"
    assert not (groups[1] & groups[2]), "Leakage detected between dev and test!"

    # Ensure all splits contain both binary classes (0 and 1)
    for s in ["train", "dev", "test"]:
        labels = set(splits[s]["label"])
        assert labels == {0, 1}, f"Split {s} is missing binary classes: {labels}"
