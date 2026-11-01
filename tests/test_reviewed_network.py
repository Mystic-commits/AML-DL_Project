import pandas as pd
import pytest
from src.causal import extract, observations, oracle, query
from src.reviewed_network import fit_reviewed_network


def inputs():
    rows = extract('Rain causes sprinkler. Rain causes wet grass. Sprinkler causes wet grass.', 'teaching').to_dict('records')
    for row in rows:
        row['approved'] = True
    assumptions = dict(reviewed_by='test reviewer', causal_rationale='Known synthetic SCM',
        observed_confounders=True, stable_mechanisms=True)
    return rows, observations(), assumptions


def test_text_review_fit_and_intervene():
    rows, data, assumptions = inputs()
    model, audit = fit_reviewed_network(rows, data, assumptions)
    assert audit['acyclic'] and len(audit['accepted']) == 3
    for value in [0,1]:
        assert query(value=value, intervention=True, network=model) == pytest.approx(oracle(value), abs=.035)


@pytest.mark.parametrize('bad', ['approval','provenance','cycle','columns','binary','missing','assumptions'])
def test_invalid_network_inputs_fail(bad):
    rows, data, assumptions = inputs()
    if bad == 'approval': rows[0]['approved'] = False
    if bad == 'provenance': rows[0].pop('document_id')
    if bad == 'cycle': rows.append(dict(rows[0], cause='wet_grass', effect='rain'))
    if bad == 'columns': data = data.drop(columns='rain')
    if bad == 'binary': data.loc[0,'rain'] = 2
    if bad == 'missing': data.loc[0,'rain'] = float('nan')
    if bad == 'assumptions': assumptions['observed_confounders'] = False
    with pytest.raises(ValueError): fit_reviewed_network(rows, data, assumptions)
