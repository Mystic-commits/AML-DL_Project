"""
Export Application Assets for Offline & Cloud Deployment.
Project 1: Neuro-Symbolic Causal Inference (Phase 1)
Team T094, Section E

Prepares lightweight, portable inference artifacts so that Streamlit
starts instantly without retraining or downloading datasets at runtime:
1. Serializes the fitted TF-IDF + Logistic Regression model to JSON (`artifacts/reasoner_snapshot.json`).
2. Exports 20 held-out Corr2Cause test examples for the interactive demo (`data/demo/reasoning_examples.csv`).
3. Generates the demo documentation (`data/demo/README.md`).
"""

import json
from pathlib import Path

import joblib
import pandas as pd

from src.runtime import export_snapshot

ROOT = Path(__file__).resolve().parents[1]


def main():
    artifacts_dir = ROOT / "artifacts"
    demo_dir = ROOT / "data/demo"
    demo_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / "reasoner.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Fitted model not found at {model_path}. Run 'python -m src.experiment' first."
        )

    print("Loading fitted reasoner pipeline...")
    fitted_model = joblib.load(model_path)

    snapshot_path = artifacts_dir / "reasoner_snapshot.json"
    print(f"Exporting portable JSON snapshot to {snapshot_path}...")
    export_snapshot(fitted_model, snapshot_path)

    test_preds_path = artifacts_dir / "test_predictions.csv"
    if not test_preds_path.exists():
        raise FileNotFoundError(
            f"Test predictions not found at {test_preds_path}. Run 'python -m src.experiment' first."
        )

    predictions_df = pd.read_csv(test_preds_path)
    # Extract 20 representative demonstration examples
    demo_examples = predictions_df.head(20)[["input", "label"]]
    demo_csv_path = demo_dir / "reasoning_examples.csv"
    demo_examples.to_csv(demo_csv_path, index=False)
    print(f"Exported 20 held-out examples to {demo_csv_path}")

    # Record provenance documentation
    manifest_path = ROOT / "data/raw/manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        dataset_name = manifest.get("dataset", "causal-nlp/corr2cause")
        revision = manifest.get("revision", "pinned")
    else:
        dataset_name = "causal-nlp/corr2cause"
        revision = "42ba12c769e11ff6427c9f52d7db58e3f9bf3e53"

    readme_content = (
        "# Reasoning demonstration examples\n\n"
        "Twenty retained held-out Corr2Cause examples for the interactive app. "
        "These are demonstration inputs, not a replacement evaluation split.\n\n"
        f"Source: https://huggingface.co/datasets/{dataset_name}\n\n"
        f"Revision: `{revision}`. Labels retain the source meaning: "
        "1 means entailed and 0 means not entailed.\n"
    )
    (demo_dir / "README.md").write_text(readme_content)
    print("Exported demonstration assets successfully.")


if __name__ == "__main__":
    main()
