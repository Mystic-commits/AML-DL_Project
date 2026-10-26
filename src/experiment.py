"""
Classical Machine Learning Baselines for Causal Text Reasoning.
Project 1: Neuro-Symbolic Causal Inference (Phase 1)
Team T094, Section E

This script:
1. Loads the Corr2Cause dataset splits (train, dev, test).
2. Groups by premise text to prevent premise leakage across splits.
3. Fits classical baselines (Majority, Naive Bayes, Logistic Regression) using train data only.
4. Selects the best performing model based on development set Macro F1.
5. Evaluates the selected model on the held-out test set with group-bootstrap confidence intervals.
6. Exports training metrics and processed splits for reproduction.
"""

import hashlib
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]


def premise(text):
    """
    Extract the premise part from a Corr2Cause formatted text.
    
    The benchmark concatenates the premise statements with the target hypothesis.
    We isolate the premise by splitting before 'Hypothesis:' or 'Conclusion:'.
    Whitespace is normalized to ensure clean hashing for premise-level grouping.
    """
    raw_premise = re.split(
        r"Hypothesis:|Conclusion:", str(text), maxsplit=1, flags=re.I
    )[0]
    # Normalize multiple whitespace characters and lower-case
    normalized = " ".join(raw_premise.strip().lower().split())
    return normalized


def load_splits():
    """
    Load raw CSV splits and enforce premise-disjoint grouping.
    
    To avoid optimistic evaluation due to duplicate or identical premises
    appearing across train, dev, and test sets, we hash the normalized premise
    and drop any examples in dev or test whose premise was already observed in train.
    """
    frames = {}
    seen_groups = set()
    audit = {}

    for split in ["train", "dev", "test"]:
        csv_path = ROOT / f"data/raw/{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Missing {csv_path}. Please run 'python scripts/fetch_data.py' first."
            )

        df = pd.read_csv(csv_path)
        df["label"] = df["label"].astype(int)

        # Compute SHA-256 hash of the cleaned premise for grouping
        df["group"] = df["input"].map(
            lambda t: hashlib.sha256(premise(t).encode("utf-8")).hexdigest()
        )

        initial_count = len(df)

        # Deduplicate exact inputs within the split
        df = df.drop_duplicates(subset=["input"])

        # Filter out premise groups seen in earlier splits to prevent leakage
        df = df[~df["group"].isin(seen_groups)].copy()
        seen_groups.update(df["group"])

        audit[split] = {
            "raw": initial_count,
            "retained": len(df),
            "removed": initial_count - len(df),
            "groups": int(df["group"].nunique()),
        }
        frames[split] = df

    return frames, audit


def scores(y_true, y_pred):
    """
    Compute binary classification metrics and macro F1 score.
    
    Given class imbalance in Corr2Cause, Macro F1 is the primary selection metric
    because it weights positive and negative classes equally.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm,
    }


def run():
    """
    Main training and evaluation routine for Phase 1.
    """
    # Ensure target output directories exist
    (ROOT / "data/processed").mkdir(parents=True, exist_ok=True)
    (ROOT / "artifacts").mkdir(parents=True, exist_ok=True)

    frames, audit = load_splits()
    train_df = frames["train"]
    dev_df = frames["dev"]
    test_df = frames["test"]

    # Define classical model pipelines
    # Single-letter variable names (A, B, C) are critical in Corr2Cause, so token_pattern retains single characters
    models = {
        "Majority baseline": Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    token_pattern=r"(?u)\b\w+\b",
                    max_features=18000,
                ),
            ),
            ("model", DummyClassifier(strategy="most_frequent")),
        ]),
        "TF-IDF + Naive Bayes": Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    token_pattern=r"(?u)\b\w+\b",
                    ngram_range=(1, 2),
                    max_features=18000,
                ),
            ),
            ("model", MultinomialNB(alpha=1.0)),
        ]),
        "TF-IDF + Logistic Regression": Pipeline([
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
        ]),
        "Unigram ablation": Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    token_pattern=r"(?u)\b\w+\b",
                    ngram_range=(1, 1),
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
        ]),
    }

    results = {}

    # Step 1: Train each model strictly on the training set
    print("Training classical models on premise-disjoint training split...")
    for name, pipeline in models.items():
        pipeline.fit(train_df["input"], train_df["label"])
        dev_preds = pipeline.predict(dev_df["input"])
        results[name] = {"dev": scores(dev_df["label"], dev_preds)}

    # Step 2: Model selection strictly on validation/dev macro F1
    # Ablations are excluded from model selection
    eligible_candidates = [m for m in models if m != "Unigram ablation"]
    winner = max(
        eligible_candidates,
        key=lambda m: results[m]["dev"]["macro_f1"],
    )
    print(f"Selected winner based on dev macro F1: {winner}")

    # Step 3: Evaluate on held-out test split
    for name, pipeline in models.items():
        test_preds = pipeline.predict(test_df["input"])
        results[name]["test"] = scores(test_df["label"], test_preds)

        if name == winner:
            out_test = test_df.copy()
            out_test["prediction"] = test_preds
            if hasattr(pipeline["model"], "predict_proba"):
                out_test["probability"] = pipeline.predict_proba(
                    test_df["input"]
                )[:, 1]
            out_test.to_csv(
                ROOT / "artifacts/test_predictions.csv", index=False
            )
            joblib.dump(pipeline, ROOT / "artifacts/reasoner.joblib")

    # Step 4: Group bootstrap on test set (resampling premise groups with replacement)
    rng = np.random.default_rng(42)
    winner_preds = models[winner].predict(test_df["input"])
    unique_groups = test_df["group"].unique()
    bootstrap_macro_f1s = []

    for _ in range(300):
        sampled_groups = rng.choice(
            unique_groups, size=len(unique_groups), replace=True
        )
        sample_indices = np.concatenate([
            np.flatnonzero(test_df["group"].to_numpy() == g)
            for g in sampled_groups
        ])
        bootstrap_score = scores(
            test_df["label"].to_numpy()[sample_indices],
            winner_preds[sample_indices],
        )["macro_f1"]
        bootstrap_macro_f1s.append(bootstrap_score)

    ci_bounds = np.quantile(bootstrap_macro_f1s, [0.025, 0.975]).tolist()

    # Step 5: Save audit metrics and processed dataset splits
    report = {
        "seed": 42,
        "selection": "highest dev macro F1; test never used for selection",
        "selected_model": winner,
        "audit": audit,
        "results": results,
        "test_macro_f1_group_bootstrap_95ci": ci_bounds,
        "limitations": [
            "Bounded prefix of training data; not a random full-corpus benchmark.",
            "Premise text grouping blocks exact premise overlap; semantic equivalence under variable renaming is not guaranteed.",
            "Corr2Cause measures reasoning entailment, not open-domain relation extraction.",
            "Classifier probabilities are not calibrated causal confidence.",
        ],
    }

    (ROOT / "artifacts/metrics.json").write_text(json.dumps(report, indent=2))

    for split_name, df_data in frames.items():
        df_data.to_csv(
            ROOT / f"data/processed/{split_name}.csv", index=False
        )

    print("\nExperiment completed successfully.")
    print(f"Winner: {winner}")
    print(f"Test Macro F1: {results[winner]['test']['macro_f1']:.3f}")
    print(f"95% Bootstrap CI: [{ci_bounds[0]:.3f}, {ci_bounds[1]:.3f}]")

    return report


if __name__ == "__main__":
    run()
