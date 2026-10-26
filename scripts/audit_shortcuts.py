"""
Shortcut & Spurious Cue Diagnostic.
Project 1: Neuro-Symbolic Causal Inference (Phase 1)
Team T094, Section E

Trains a hypothesis-only control model to measure how much signal the classifier
picks up strictly from superficial phrasing of the hypothesis, without access
to the causal premise text.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.experiment import load_splits, scores

ROOT = Path(__file__).resolve().parents[1]


def extract_hypothesis(text):
    """
    Isolate the hypothesis portion of the prompt.
    """
    parts = re.split(r"Hypothesis:|Conclusion:", str(text), maxsplit=1, flags=re.I)
    if len(parts) > 1:
        return parts[1].strip()
    return text.strip()


def run_audit():
    (ROOT / "artifacts").mkdir(parents=True, exist_ok=True)

    print("Loading premise-disjoint splits for shortcut diagnostic...")
    frames, _ = load_splits()
    train_df = frames["train"]
    dev_df = frames["dev"]
    test_df = frames["test"]

    # Extract hypothesis only
    train_hyp = train_df["input"].map(extract_hypothesis)
    dev_hyp = dev_df["input"].map(extract_hypothesis)
    test_hyp = test_df["input"].map(extract_hypothesis)

    # Identical architecture to the selected reasoning model
    control_model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                token_pattern=r"(?u)\b\w+\b",
                ngram_range=(1, 2),
                max_features=18000,
                sublinear_tf=True,
            ),
        ),
        (
            "model",
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=1500,
                random_state=42,
            ),
        ),
    ])

    print("Fitting hypothesis-only control model...")
    control_model.fit(train_hyp, train_df["label"])

    dev_preds = control_model.predict(dev_hyp)
    test_preds = control_model.predict(test_hyp)

    dev_scores = scores(dev_df["label"], dev_preds)
    test_scores = scores(test_df["label"], test_preds)

    full_metrics = json.loads((ROOT / "artifacts/metrics.json").read_text())
    full_test_f1 = full_metrics["results"]["TF-IDF + Logistic Regression"]["test"]["macro_f1"]

    audit_payload = {
        "description": "Hypothesis-only diagnostic measuring dataset shortcut artifacts",
        "results": {
            "dev": dev_scores,
            "test": test_scores,
        },
        "full_input_test_macro_f1": full_test_f1,
        "hypothesis_only_test_macro_f1": test_scores["macro_f1"],
        "interpretation": (
            f"The hypothesis-only model achieves {test_scores['macro_f1']:.3f} test macro F1 "
            f"without observing the premise, compared to {full_test_f1:.3f} for the full input model. "
            "This indicates that synthetic benchmark templates contain superficial wording "
            "and label-prior shortcuts that a bag-of-words classifier exploits without formal reasoning."
        ),
    }

    output_path = ROOT / "artifacts/shortcut_audit.json"
    output_path.write_text(json.dumps(audit_payload, indent=2))
    print(f"Saved shortcut diagnostic to {output_path}")
    print(audit_payload["interpretation"])
    return audit_payload


if __name__ == "__main__":
    run_audit()
