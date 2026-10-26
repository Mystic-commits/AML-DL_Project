"""
Validation of individual system components:
1. SCM parameter extraction and synthetic dataset export
2. Lexical causal relation extractor against regression fixtures
3. DAG construction and cycle rejection
4. Pearl's do-calculus intervention query accuracy against the closed-form oracle

Project 1: Neuro-Symbolic Causal Inference (Phase 1)
Team T094, Section E
"""

import json
from pathlib import Path

from src.causal import (
    build_graph,
    extract,
    fitted_network,
    observations,
    oracle,
    query,
)

ROOT = Path(__file__).resolve().parents[1]


def main():
    # Ensure target directories exist before saving
    (ROOT / "data/processed").mkdir(parents=True, exist_ok=True)
    (ROOT / "artifacts").mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 1. Export synthetic observations and serialized CPD tables
    # -------------------------------------------------------------
    print("Generating synthetic observation dataset (6,000 samples)...")
    synthetic_df = observations(n=6000, seed=42)
    synthetic_df.to_csv(
        ROOT / "data/processed/synthetic_observations.csv", index=False
    )

    print("Fitting discrete Bayesian network using BDeu prior...")
    bayesian_model = fitted_network()
    cpds = {
        cpd.variable: {
            "variables": list(cpd.variables),
            "values": cpd.values.tolist(),
        }
        for cpd in bayesian_model.get_cpds()
    }

    bayesian_params = {
        "seed": 42,
        "rows": 6000,
        "prior": "BDeu",
        "equivalent_sample_size": 5,
        "edges": list(bayesian_model.edges()),
        "cpds": cpds,
    }
    (ROOT / "artifacts/bayesian_parameters.json").write_text(
        json.dumps(bayesian_params, indent=2)
    )

    # -------------------------------------------------------------
    # 2. Validate relation extractor against regression documents
    # -------------------------------------------------------------
    print("Evaluating lexical extractor on 16 authored regression fixtures...")
    fixture_path = ROOT / "data/extraction_gold.json"
    fixtures = json.loads(fixture_path.read_text())

    tp = 0
    fp = 0
    fn = 0
    doc_details = []

    for doc in fixtures:
        extracted_records = extract(doc["text"], doc["id"]).to_dict("records")
        predicted_edges = {
            (rec["cause"], rec["effect"]) for rec in extracted_records
        }
        gold_edges = {tuple(edge) for edge in doc["edges"]}

        true_positives = len(predicted_edges & gold_edges)
        false_positives = len(predicted_edges - gold_edges)
        false_negatives = len(gold_edges - predicted_edges)

        tp += true_positives
        fp += false_positives
        fn += false_negatives

        doc_details.append({
            "id": doc["id"],
            "predicted": sorted(predicted_edges),
            "gold": sorted(gold_edges),
        })

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # -------------------------------------------------------------
    # 3. Validate DAG building and cycle detection
    # -------------------------------------------------------------
    print("Validating DAG cycle rejection on 4-edge sample text...")
    sample_text = (
        "Rain causes sprinkler. Rain causes wet grass. "
        "Sprinkler causes wet grass. Wet grass causes rain."
    )
    raw_edges = extract(sample_text).to_dict("records")
    graph, accepted_edges, rejected_edges = build_graph(raw_edges)

    gold_dag_edges = {
        ("rain", "sprinkler"),
        ("rain", "wet_grass"),
        ("sprinkler", "wet_grass"),
    }
    pred_dag_edges = {
        (rec["cause"], rec["effect"]) for rec in accepted_edges
    }

    graph_precision = (
        len(pred_dag_edges & gold_dag_edges) / len(pred_dag_edges)
        if pred_dag_edges
        else 0.0
    )
    graph_recall = (
        len(pred_dag_edges & gold_dag_edges) / len(gold_dag_edges)
        if gold_dag_edges
        else 0.0
    )

    # -------------------------------------------------------------
    # 4. Validate do-calculus queries against oracle values
    # -------------------------------------------------------------
    print("Comparing Bayesian intervention queries against mathematical oracle...")
    intervention_results = []
    for s_val in [0, 1]:
        p_obs = query(variable="sprinkler", value=s_val, network=bayesian_model)
        p_do = query(
            variable="sprinkler",
            value=s_val,
            intervention=True,
            network=bayesian_model,
        )
        p_oracle = oracle(s_val)
        abs_err = abs(p_do - p_oracle)

        intervention_results.append({
            "sprinkler": s_val,
            "observed": float(p_obs),
            "estimated_do": float(p_do),
            "oracle_do": float(p_oracle),
            "absolute_error": float(abs_err),
        })

    validation_report = {
        "fixture_notice": (
            "16 manually authored regression documents, not an independently annotated "
            "held-out corpus. Metrics describe supported-pattern coverage only."
        ),
        "extraction": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        },
        "graph": {
            "precision": float(graph_precision),
            "recall": float(graph_recall),
            "rejected": rejected_edges,
        },
        "interventions": intervention_results,
        "documents": doc_details,
    }

    output_path = ROOT / "artifacts/component_validation.json"
    output_path.write_text(json.dumps(validation_report, indent=2))
    print(f"Validation report saved to {output_path}")
    print(
        f"Extraction: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f} (TP={tp}, FP={fp}, FN={fn})"
    )
    print(
        f"Intervention errors vs oracle: {[f'{r['absolute_error']:.4f}' for r in intervention_results]}"
    )


if __name__ == "__main__":
    main()
