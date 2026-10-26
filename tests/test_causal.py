"""
Unit tests for causal relation extraction, DAG construction, and Bayesian interventions.
Project 1: Neuro-Symbolic Causal Inference (Phase 1)
Team T094, Section E
"""

import networkx as nx
import pytest
from src.causal import build_graph, extract, oracle, query


def test_review_rejects_cycles_and_self_loops():
    """
    Ensure DAG builder rejects self-loops and directed cycles
    while accepting valid non-cyclical edges.
    """
    text = "Rain causes wet grass. Wet grass causes rain. Rain causes rain."
    extracted_edges = extract(text).to_dict("records")
    graph, accepted, rejected = build_graph(extracted_edges)

    assert nx.is_directed_acyclic_graph(graph), "Constructed graph contains cycles!"
    assert len(accepted) == 1, f"Expected 1 accepted edge, got {len(accepted)}"

    reasons = {rec["reason"] for rec in rejected}
    expected_reasons = {"self loop", "would create a cycle"}
    assert reasons == expected_reasons, f"Expected {expected_reasons}, got {reasons}"


def test_uncertain_and_negative_statements_are_not_claimed():
    """
    Ensure the regex extractor conservatively ignores statements with
    modal verbs ('may', 'might') or negation ('not').
    """
    negated_text = "Rain may cause flooding. Rain does not cause drought."
    extracted = extract(negated_text)
    assert extracted.empty, "Extractor falsely extracted from uncertain/negated text!"


def test_intervention_matches_generating_process():
    """
    Verify that pgmpy's do-calculus estimation matches the mathematical oracle
    within sampling tolerance (0.035) and differs from passive observation.
    """
    for value in [0, 1]:
        p_do = query(variable="sprinkler", value=value, intervention=True)
        p_oracle = oracle(value)
        assert p_do == pytest.approx(p_oracle, abs=0.035), (
            f"Intervention estimate {p_do:.3f} diverged from oracle {p_oracle:.3f}"
        )

    # Confounding check: P(wet | sprinkler=1) != P(wet | do(sprinkler=1))
    p_obs = query(variable="sprinkler", value=1, intervention=False)
    p_do = query(variable="sprinkler", value=1, intervention=True)
    assert abs(p_obs - p_do) > 0.03, "Observation and intervention should differ under confounding!"
