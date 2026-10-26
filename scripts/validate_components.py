"""Small authored regression fixture; not a generalization benchmark."""
import json
from pathlib import Path
from src.causal import extract,build_graph,query,oracle,observations,fitted_network
ROOT=Path(__file__).resolve().parents[1]
def main():
    observations().to_csv(ROOT/'data/processed/synthetic_observations.csv',index=False)
    model=fitted_network()
    cpds={cpd.variable:{'variables':list(cpd.variables),'values':cpd.values.tolist()} for cpd in model.get_cpds()}
    (ROOT/'artifacts/bayesian_parameters.json').write_text(json.dumps({'seed':42,'rows':6000,'prior':'BDeu','equivalent_sample_size':5,'edges':list(model.edges()),'cpds':cpds},indent=2))
    fixture=json.loads((ROOT/'data/extraction_gold.json').read_text());tp=fp=fn=0;details=[]
    for doc in fixture:
        predicted={(r['cause'],r['effect']) for r in extract(doc['text'],doc['id']).to_dict('records')}
        gold={tuple(e) for e in doc['edges']}
        tp+=len(predicted&gold);fp+=len(predicted-gold);fn+=len(gold-predicted)
        details.append({'id':doc['id'],'predicted':sorted(predicted),'gold':sorted(gold)})
    precision=tp/(tp+fp) if tp+fp else 0;recall=tp/(tp+fn) if tp+fn else 0
    _,accepted,rejected=build_graph(extract('Rain causes sprinkler. Rain causes wet grass. Sprinkler causes wet grass. Wet grass causes rain.').to_dict('records'))
    gold_edges={('rain','sprinkler'),('rain','wet_grass'),('sprinkler','wet_grass')}
    pred_edges={(r['cause'],r['effect']) for r in accepted}
    report={'fixture_notice':'16 manually authored regression documents, not an independently annotated held-out corpus. Metrics describe supported-pattern coverage only.','extraction':{'tp':tp,'fp':fp,'fn':fn,'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall)},'graph':{'precision':len(pred_edges&gold_edges)/len(pred_edges),'recall':len(pred_edges&gold_edges)/len(gold_edges),'rejected':rejected},'interventions':[{'sprinkler':s,'observed':query(value=s),'estimated_do':query(value=s,intervention=True),'oracle_do':oracle(s),'absolute_error':abs(query(value=s,intervention=True)-oracle(s))} for s in [0,1]],'documents':details}
    (ROOT/'artifacts/component_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
