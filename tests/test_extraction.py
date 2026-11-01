import pandas as pd
import pytest
from src.extraction import lexical_label, relation_scores, prepare, features


def test_direction_and_negative_examples():
    assert lexical_label('<e1>Heat</e1> causes <e2>evaporation</e2>.') == 1
    assert lexical_label('<e1>Flooding</e1> is caused by <e2>rain</e2>.') == 2
    assert lexical_label('<e2>Heat</e2> causes <e1>evaporation</e1>.') == 2
    assert lexical_label('<e1>Heat</e1> may cause <e2>evaporation</e2>.') == 0
    assert lexical_label('<e1>Heat</e1> correlates with <e2>evaporation</e2>.') == 0


def test_reversed_edge_counts_as_false_positive_and_false_negative():
    score = relation_scores([1,2,0], [2,2,1])
    assert (score['tp'], score['fp'], score['fn']) == (1,2,1)
    assert score['edge_precision'] == pytest.approx(1/3)
    assert score['edge_recall'] == .5


def test_schema_and_conflicting_labels():
    with pytest.raises(ValueError): prepare(pd.DataFrame({'sentence':['x'],'relation':[0]}))
    df = pd.DataFrame({'sentence':['<e1>x</e1> causes <e2>y</e2>.']*2, 'relation':[0,1]})
    with pytest.raises(ValueError): prepare(df)
    assert 'secret' not in features('<e1>secret</e1> causes <e2>outcome</e2>')


def test_portable_multiclass_matches_fitted_model(tmp_path):
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from src.extraction_runtime import export_model, FrozenExtractor
    text = ['rain causes flooding', 'traffic due to rain', 'water in glass']*3
    model = Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)),
        ('classifier',LogisticRegression())]).fit(text,[1,2,0]*3)
    path = tmp_path/'model.json'
    export_model(model,path)
    assert np.array_equal(model.predict(text), FrozenExtractor(path).predict(text))
