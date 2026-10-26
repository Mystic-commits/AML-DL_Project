"""
Post-Experiment Analysis & EDA Generation.
Project 1: Neuro-Symbolic Causal Inference (Phase 1)
Team T094, Section E

This script generates:
1. EDA summary table and split length / class distribution figures (`artifacts/eda_summary.csv`, `artifacts/eda.png`).
2. Error analysis for misclassified test examples (`artifacts/error_examples.csv`).
3. Sliced performance metrics by causal graph template (`artifacts/template_metrics.csv`).
4. Most influential positive and negative lexical features (`artifacts/feature_weights.csv`).
"""

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiment import load_splits, scores

ROOT = Path(__file__).resolve().parents[1]


def main():
    # Ensure artifacts directory exists
    (ROOT / "artifacts").mkdir(parents=True, exist_ok=True)

    print("Loading dataset splits for exploratory data analysis...")
    frames, _ = load_splits()

    # -------------------------------------------------------------
    # 1. Dataset statistics and word-length distributions
    # -------------------------------------------------------------
    records = []
    for split_name, df_data in frames.items():
        lengths = df_data["input"].str.split().str.len()
        records.append({
            "split": split_name,
            "rows": len(df_data),
            "positive": int(df_data["label"].sum()),
            "negative": int((df_data["label"] == 0).sum()),
            "positive_rate": float(df_data["label"].mean()),
            "median_words": float(lengths.median()),
            "max_words": int(lengths.max()),
            "missing_input": int(df_data["input"].isna().sum()),
        })

    summary_df = pd.DataFrame(records)
    summary_df.to_csv(ROOT / "artifacts/eda_summary.csv", index=False)
    print("EDA Summary:")
    print(summary_df.to_string(index=False))

    # -------------------------------------------------------------
    # 2. Generate clean, high-contrast EDA visualization figure
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor="#ffffff")
    for ax in axes:
        ax.set_facecolor("#ffffff")
        ax.tick_params(colors="#1a1a1a")
        for spine in ax.spines.values():
            spine.set_color("#1a1a1a")

    # Panel A: Input length distribution per split
    for split_name, df_data in frames.items():
        word_counts = df_data["input"].str.split().str.len()
        axes[0].hist(
            word_counts,
            bins=25,
            alpha=0.45,
            label=split_name,
            density=True,
        )
    axes[0].set_xlabel("Words per example", color="#1a1a1a", fontsize=10)
    axes[0].set_ylabel("Density", color="#1a1a1a", fontsize=10)
    axes[0].set_title("Input Token Length Distribution", color="#1a1a1a", fontsize=11, fontweight="bold")
    axes[0].legend(frameon=True, facecolor="#ffffff", edgecolor="#1a1a1a")

    # Panel B: Class imbalance across splits (stacked bar)
    imbalance_df = summary_df.set_index("split")[["negative", "positive"]]
    imbalance_df.plot.bar(
        stacked=True,
        ax=axes[1],
        color=["#222222", "#888888"],
        edgecolor="#1a1a1a",
    )
    axes[1].set_title("Class Imbalance (0: Not Entailed, 1: Entailed)", color="#1a1a1a", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Number of Examples", color="#1a1a1a", fontsize=10)
    axes[1].set_xlabel("Split", color="#1a1a1a", fontsize=10)
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(["Negative (0)", "Positive (1)"], frameon=True, facecolor="#ffffff", edgecolor="#1a1a1a")

    fig.tight_layout()
    fig_path = ROOT / "artifacts/eda.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved EDA visualization to {fig_path}")

    # -------------------------------------------------------------
    # 3. Error analysis on test predictions
    # -------------------------------------------------------------
    preds_path = ROOT / "artifacts/test_predictions.csv"
    if preds_path.exists():
        predictions = pd.read_csv(preds_path)
        errors = predictions[predictions["label"] != predictions["prediction"]]
        # Sort false positives by high probability (overconfident errors)
        errors_sorted = errors.sort_values(by="probability", ascending=False)
        errors_path = ROOT / "artifacts/error_examples.csv"
        errors_sorted.to_csv(errors_path, index=False)
        print(f"Exported {len(errors_sorted)} error examples to {errors_path}")

        # Template slices: breakdown by structural template
        if "template" in predictions.columns:
            slices = []
            for template_name, group in predictions.groupby("template"):
                template_scores = scores(group["label"], group["prediction"])
                template_scores.pop("confusion_matrix", None)
                slices.append({
                    "template": template_name,
                    "n": len(group),
                    **template_scores,
                })
            slices_df = pd.DataFrame(slices)
            slices_path = ROOT / "artifacts/template_metrics.csv"
            slices_df.to_csv(slices_path, index=False)
            print(f"Saved template-level breakdown to {slices_path}")

    # -------------------------------------------------------------
    # 4. Feature weights inspection (interpretable linear model)
    # -------------------------------------------------------------
    model_path = ROOT / "artifacts/reasoner.joblib"
    if model_path.exists():
        pipeline = joblib.load(model_path)
        if hasattr(pipeline["model"], "coef_"):
            feature_names = pipeline["tfidf"].get_feature_names_out()
            weights = pipeline["model"].coef_[0]
            sorted_indices = np.argsort(weights)
            # Pick top 15 most negative and top 15 most positive n-grams
            selected_indices = np.r_[sorted_indices[:15], sorted_indices[-15:]]
            weights_df = pd.DataFrame({
                "feature": feature_names[selected_indices],
                "weight": weights[selected_indices],
            })
            weights_path = ROOT / "artifacts/feature_weights.csv"
            weights_df.to_csv(weights_path, index=False)
            print(f"Saved top feature weights to {weights_path}")


if __name__ == "__main__":
    main()
