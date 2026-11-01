"""Fit a reviewed DAG to matching binary observations and export its assumptions."""
import math

import networkx as nx
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.parameter_estimator import DiscreteBayesianEstimator
from src.causal import build_graph


def fit_reviewed_network(records, data, assumptions, equivalent_sample_size=5):
    """Requires explicit approval, provenance and caller's causal assumptions.

    Observations estimate CPDs; approval alone does not identify causal effects.
    Binary variables and complete observations keep the teaching contract auditable.
    """
    required = ['reviewed_by', 'causal_rationale', 'observed_confounders', 'stable_mechanisms']
    if any(not assumptions.get(k) for k in required):
        raise ValueError('Record reviewer, causal rationale, observed confounders and stable mechanisms')
    if not records or any(r.get('approved') is not True for r in records):
        raise ValueError('Every submitted edge requires explicit approval')
    for row in records:
        if not row.get('document_id') or not row.get('sentence') or not row.get('sentence_id'):
            raise ValueError('Every edge requires document and sentence provenance')
    graph, accepted, rejected = build_graph(records)
    if rejected:
        raise ValueError(f'Resolve invalid or duplicate edges before fitting: {rejected}')
    nodes = list(graph.nodes)
    if len(data) < 2 or not set(nodes).issubset(data.columns) or data.columns.duplicated().any():
        raise ValueError('Provide at least two rows and unique columns for every graph node')
    used = data[nodes]
    if used.isna().any().any() or not used.isin([0,1]).all().all():
        raise ValueError('This teaching model accepts complete binary 0/1 observations only')
    if not math.isfinite(equivalent_sample_size) or equivalent_sample_size <= 0:
        raise ValueError('Equivalent sample size must be positive and finite')
    model = DiscreteBayesianNetwork(graph.edges)
    model.fit(used, estimator=DiscreteBayesianEstimator(prior_type='BDeu',
        equivalent_sample_size=equivalent_sample_size, state_names={n:[0,1] for n in nodes}))
    model.check_model()
    audit = {'accepted': accepted, 'rejected': rejected, 'assumptions': assumptions,
        'rows': len(used), 'edges': list(graph.edges), 'acyclic': nx.is_directed_acyclic_graph(graph),
        'prior': 'BDeu', 'equivalent_sample_size': equivalent_sample_size,
        'cpds': {c.variable: {'variables': list(c.variables), 'values': c.values.tolist()} for c in model.get_cpds()}}
    return model, audit
