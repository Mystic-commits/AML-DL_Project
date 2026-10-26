import networkx as nx
import pytest
from src.causal import extract,build_graph,query,oracle

def test_review_rejects_cycles_and_self_loops():
    rows=extract('Rain causes wet grass. Wet grass causes rain. Rain causes rain.').to_dict('records')
    graph,accepted,rejected=build_graph(rows)
    assert nx.is_directed_acyclic_graph(graph)
    assert len(accepted)==1 and {r['reason'] for r in rejected}=={'self loop','would create a cycle'}

def test_uncertain_and_negative_statements_are_not_claimed():
    assert extract('Rain may cause flooding. Rain does not cause drought.').empty

def test_intervention_matches_generating_process():
    for value in [0,1]:
        assert query(value=value,intervention=True)==pytest.approx(oracle(value),abs=.035)
    assert abs(query(value=1)-query(value=1,intervention=True))>.03
