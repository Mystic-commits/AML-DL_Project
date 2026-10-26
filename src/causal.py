"""
Causal extraction, graph validation, and Bayesian intervention queries.

This module handles three things:
1. Extracting cause-effect pairs from text using simple regex patterns
2. Building a DAG with cycle/self-loop rejection
3. Running observational vs interventional queries on a synthetic Bayesian network
"""

import re
import copy

import networkx as nx
import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.parameter_estimator import DiscreteBayesianEstimator
from pgmpy.inference import VariableElimination


# known entity aliases so "rainfall" and "rain" map to the same node
ALIASES = {
    "rainfall": "rain",
    "rain": "rain",
    "sprinklers": "sprinkler",
    "sprinkler": "sprinkler",
    "wet grass": "wet_grass",
    "wet_grass": "wet_grass",
}

# regex for simple causal statements like "X causes Y" or "X leads to Y"
PATTERN = re.compile(
    r"(?P<cause>[a-z_ ]+?)\s+(?:causes?|leads? to|results? in)\s+(?P<effect>[a-z_ ]+)",
    re.IGNORECASE,
)


def canonical(text):
    """Normalize an entity name by stripping articles and applying aliases."""
    key = re.sub(r"^(the|a|an)\s+", "", text.lower().strip())
    return ALIASES.get(key, key.replace(" ", "_"))


def extract(text, document_id="interactive"):
    """
    Pull cause-effect pairs from text using supported lexical patterns.
    
    Only handles simple affirmative sentences. Skips anything with
    negation or uncertainty words (not, may, might, could, if).
    Returns a dataframe with provenance info for each candidate edge.
    """
    rows = []

    for i, sentence in enumerate(re.split(r"[.!?\n]+", text)):
        sentence = sentence.strip()
        if not sentence:
            continue

        match = PATTERN.fullmatch(sentence)

        # skip uncertain or negated statements
        has_uncertainty = re.search(r"\b(not|may|might|could|if)\b", sentence, re.I)

        if match and not has_uncertainty:
            rows.append({
                "cause": canonical(match["cause"]),
                "effect": canonical(match["effect"]),
                "document_id": document_id,
                "sentence_id": i + 1,
                "sentence": sentence,
                "method": "explicit lexical pattern",
            })

    columns = ["cause", "effect", "document_id", "sentence_id", "sentence", "method"]
    return pd.DataFrame(rows, columns=columns)


def build_graph(records):
    """
    Build a directed acyclic graph from a list of edge records.
    
    Greedy validation: edges are added in source order. We reject:
    - self loops (cause == effect)
    - duplicate edges already in the graph
    - edges that would create a directed cycle
    
    Returns the graph, accepted edges list, and rejected edges list.
    """
    graph = nx.DiGraph()
    accepted = []
    rejected = []

    for row in records:
        cause = row["cause"]
        effect = row["effect"]
        reason = None

        if cause == effect:
            reason = "self loop"
        elif graph.has_edge(cause, effect):
            reason = "duplicate edge"
        elif cause in graph and effect in graph and nx.has_path(graph, effect, cause):
            reason = "would create a cycle"

        if reason:
            rejected.append(dict(row, reason=reason))
        else:
            graph.add_edge(cause, effect)
            accepted.append(row)

    return graph, accepted, rejected


def observations(n=6000, seed=42):
    """
    Generate synthetic observations from the Rain/Sprinkler/Wet Grass system.
    
    Ground truth generating process:
    - P(Rain=1) = 0.3
    - P(Sprinkler=1 | Rain=1) = 0.05, P(Sprinkler=1 | Rain=0) = 0.6
    - P(Wet=1) depends on both Rain and Sprinkler (see conditional table below)
    
    The rain-sprinkler correlation creates confounding that makes
    observation differ from intervention.
    """
    rng = np.random.default_rng(seed)

    rain = rng.binomial(1, 0.3, n)
    sprinkler = rng.binomial(1, np.where(rain, 0.05, 0.6))

    # wet grass probability conditional on rain and sprinkler
    wet_prob = np.where(
        (rain == 1) & (sprinkler == 1), 0.99,
        np.where(rain == 1, 0.9,
                 np.where(sprinkler == 1, 0.8, 0.02))
    )
    wet = rng.binomial(1, wet_prob)

    return pd.DataFrame({
        "rain": rain,
        "sprinkler": sprinkler,
        "wet_grass": wet,
    })


def fitted_network():
    """
    Fit a Bayesian network to synthetic observations.
    
    Structure: Rain → Sprinkler, Rain → Wet Grass, Sprinkler → Wet Grass
    Uses BDeu prior with equivalent sample size 5 for smoothing.
    """
    edges = [
        ("rain", "sprinkler"),
        ("rain", "wet_grass"),
        ("sprinkler", "wet_grass"),
    ]
    model = DiscreteBayesianNetwork(edges)

    state_names = {k: [0, 1] for k in ["rain", "sprinkler", "wet_grass"]}
    model.fit(
        observations(),
        estimator=DiscreteBayesianEstimator(
            prior_type="BDeu",
            equivalent_sample_size=5,
            state_names=state_names,
        ),
    )
    model.check_model()
    return model


def query(target="wet_grass", variable="sprinkler", value=1,
          intervention=False, network=None):
    """
    Compute P(target | variable=value) or P(target | do(variable=value)).
    
    For interventional queries, we remove incoming edges to the variable
    (Pearl's do-operator) and then condition on its fixed state.
    
    If a pre-fitted network is passed, uses a copy of it instead of
    refitting from scratch every time.
    """
    if target == variable:
        raise ValueError("Select different query and evidence variables.")

    if network is not None:
        model = copy.deepcopy(network)
    else:
        model = fitted_network()

    if intervention:
        model = model.do([variable], inplace=False)

    result = VariableElimination(model).query(
        [target],
        evidence={variable: value},
        show_progress=False,
    )
    return float(result.values[1])


def oracle(s):
    """
    Exact mathematical answer for P(Wet=1 | do(Sprinkler=s)).
    
    Uses the generating process directly:
    P(W=1|do(S=s)) = sum_r P(W=1|r,s) * P(r)
    where P(R=1) = 0.3
    """
    return 0.3 * (0.99 if s else 0.9) + 0.7 * (0.8 if s else 0.02)
