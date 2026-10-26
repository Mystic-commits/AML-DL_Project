"""Evidence-preserving rule extraction and a separately identified synthetic SCM."""
import re
import networkx as nx
import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.parameter_estimator import DiscreteBayesianEstimator
from pgmpy.inference import VariableElimination
ALIASES={'rainfall':'rain','rain':'rain','sprinklers':'sprinkler','sprinkler':'sprinkler','wet grass':'wet_grass','wet_grass':'wet_grass'}
PATTERN=re.compile(r'(?P<cause>[a-z_ ]+?)\s+(?:causes?|leads? to|results? in)\s+(?P<effect>[a-z_ ]+)',re.I)
def canonical(text):
    key=re.sub(r'^(the|a|an)\s+','',text.lower().strip())
    return ALIASES.get(key,key.replace(' ','_'))
def extract(text,document_id='interactive'):
    rows=[]
    for i,sentence in enumerate(re.split(r'[.!?\n]+',text)):
        sentence=sentence.strip()
        if not sentence:continue
        match=PATTERN.fullmatch(sentence)
        if match and not re.search(r'\b(not|may|might|could|if)\b',sentence,re.I):
            rows.append({'cause':canonical(match['cause']),'effect':canonical(match['effect']),'document_id':document_id,'sentence_id':i+1,'sentence':sentence,'method':'explicit lexical pattern'})
    return pd.DataFrame(rows,columns=['cause','effect','document_id','sentence_id','sentence','method'])
def build_graph(records):
    graph=nx.DiGraph();accepted=[];rejected=[]
    for row in records:
        a,b=row['cause'],row['effect']
        reason=None
        if a==b:reason='self loop'
        elif graph.has_edge(a,b):reason='duplicate edge'
        elif a in graph and b in graph and nx.has_path(graph,b,a):reason='would create a cycle'
        if reason:rejected.append(dict(row,reason=reason))
        else:graph.add_edge(a,b);accepted.append(row)
    return graph,accepted,rejected
def observations(n=6000,seed=42):
    rng=np.random.default_rng(seed)
    rain=rng.binomial(1,.3,n)
    sprinkler=rng.binomial(1,np.where(rain,.05,.6))
    wet=rng.binomial(1,np.where((rain==1)&(sprinkler==1),.99,np.where(rain==1,.9,np.where(sprinkler==1,.8,.02))))
    return pd.DataFrame({'rain':rain,'sprinkler':sprinkler,'wet_grass':wet})
def fitted_network():
    model=DiscreteBayesianNetwork([('rain','sprinkler'),('rain','wet_grass'),('sprinkler','wet_grass')])
    model.fit(observations(),estimator=DiscreteBayesianEstimator(prior_type='BDeu',equivalent_sample_size=5,state_names={k:[0,1] for k in ['rain','sprinkler','wet_grass']}))
    model.check_model()
    return model
def query(target='wet_grass',variable='sprinkler',value=1,intervention=False):
    if target==variable:raise ValueError('Select different query and evidence variables.')
    model=fitted_network()
    if intervention:
        model=model.do([variable],inplace=False)
    return float(VariableElimination(model).query([target],evidence={variable:value},show_progress=False).values[1])
def oracle(s):
    return .3*(.99 if s else .9)+.7*(.8 if s else .02)
