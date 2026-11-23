"""One frozen-protocol held-out evaluation of neural variants and the deployed export."""
import argparse
import gc
import json
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from scripts.verify_neural_run import verify
from src.causal import build_graph, observations, oracle, query
from src.extraction import features, relation_scores
from src.extraction_runtime import FrozenExtractor
from src.hybrid import candidate_record
from src.neural import choose_device
from src.neural_data import file_sha
from src.neural_runtime import OnnxExtractor
from src.neural_training import evaluate, make_loader
from src.reviewed_network import fit_reviewed_network

ROOT = Path(__file__).resolve().parents[1]


def edge_f1(gold,pred):
    tp=np.sum((gold==pred)&(gold!=0));fp=np.sum((pred!=0)&(gold!=pred));fn=np.sum((gold!=0)&(gold!=pred))
    return float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.


def graph_case(model):
    sentences=['<e1>Rain</e1> causes <e2>sprinkler</e2>.','<e1>Rain</e1> causes <e2>wet grass</e2>.',
               '<e1>Sprinkler</e1> causes <e2>wet grass</e2>.','<e1>Wet grass</e1> causes <e2>rain</e2>.']
    probs=model.predict_proba(sentences)
    rows=[candidate_record(t,int(p.argmax()),p,'controlled_synthetic_case',i+1) for i,(t,p) in enumerate(zip(sentences,probs))]
    rows=[r for r in rows if r]
    graph,accepted,rejected=build_graph(rows)
    gold={('rain','sprinkler'),('rain','wet_grass'),('sprinkler','wet_grass')}
    raw={(r['cause'],r['effect']) for r in rows}; kept=set(graph.edges())
    def score(edges):
        tp=len(edges&gold);p=tp/len(edges) if edges else 0.;r=tp/len(gold)
        return {'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0.}
    result={'scope':'Single controlled synthetic case; not a real-world graph benchmark',
        'review':'Scripted known-generator review for evaluation, not a claim of independent human annotation',
        'raw_candidates':rows,'accepted':accepted,'rejected':rejected,
        'raw_edge_metrics':score(raw),'validated_edge_metrics':score(kept),
        'raw_acyclic':nx.is_directed_acyclic_graph(nx.DiGraph(list(raw))),
        'validated_acyclic':nx.is_directed_acyclic_graph(graph)}
    if kept!=gold:
        raise ValueError('Neural controlled case did not recover the supplied teaching structure')
    network,audit=fit_reviewed_network([dict(r,approved=True) for r in accepted],observations(),dict(
        reviewed_by='Scripted known-generator reviewer', causal_rationale='Known synthetic generating graph; evaluation case only',
        observed_confounders=True,stable_mechanisms=True))
    result['queries']=[{'sprinkler':v,'observed':query(variable='sprinkler',value=v,network=network),
        'intervened':query(variable='sprinkler',value=v,intervention=True,network=network),'oracle':oracle(v)} for v in [0,1]]
    result['maximum_oracle_error']=max(abs(q['intervened']-q['oracle']) for q in result['queries'])
    result['network_audit']=audit
    return result


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',nargs='+',type=Path,required=True)
    parser.add_argument('--onnx',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Frozen evaluation output already exists')
    protocol=json.loads((ROOT/'configs/phase2_final_evaluation.json').read_text())
    audits=[verify(p) for p in args.runs]
    reports=[json.loads((p/'run.json').read_text()) for p in args.runs]
    required={('fine_tuned_weighted',42),('fine_tuned_weighted',17),('fine_tuned_weighted',2026),('frozen_encoder_weighted',42),('fine_tuned_unweighted',42)}
    if {(r['variant'],r['seed']) for r in reports}!=required:raise ValueError('Expected the five predeclared runs')
    audit=json.loads((ROOT/'artifacts/phase2/data_audit.json').read_text())
    path=ROOT/'data/processed/phase2_extraction/test.csv'
    if file_sha(path)!=audit['local_files_sha256']['test.csv']:raise ValueError('Test source changed')
    test=pd.read_csv(path);gold=test.label.to_numpy()
    historical=ROOT/'artifacts/runs/phase1_extraction_verified_20261101'
    baseline=pd.read_csv(historical/'predictions.csv')
    if not baseline[['group','label']].equals(test[['group','label']]):raise ValueError('Baseline test alignment mismatch')
    ml=FrozenExtractor(historical/'model_snapshot.json').predict(test.sentence.map(features).tolist())
    if not np.array_equal(ml,baseline.tfidf_lr):raise ValueError('Historical classifier reproduction mismatch')
    predictions={name:baseline[name].to_numpy() for name in ['majority','lexical','tfidf_lr','entity_only_ablation']}
    metrics={name:relation_scores(gold,pred) for name,pred in predictions.items()}
    selected={};device=choose_device()
    args.output.mkdir(parents=True)
    for folder,report in zip(args.runs,reports):
        name=f'{report["variant"]}_seed{report["seed"]}'
        tok=AutoTokenizer.from_pretrained(folder/'checkpoint',local_files_only=True,trust_remote_code=False)
        model=AutoModelForSequenceClassification.from_pretrained(folder/'checkpoint',local_files_only=True,
            trust_remote_code=False,use_safetensors=True,attn_implementation='eager').to(device)
        loader=make_loader(test,tok,report['config']['max_length'],32,False,report['seed'])
        score,probs=evaluate(model,loader,device)
        predictions[name]=probs.argmax(axis=1);metrics[name]=score
        selected[name]={'selected_epoch':report['selected_epoch'],'development_edge_f1':report['selected_development_metrics']['edge_f1'],
            'checkpoint_sha256':file_sha(folder/'checkpoint/model.safetensors'),'seed':report['seed'],'variant':report['variant']}
        np.save(args.output/f'{name}_probabilities.npy',probs)
        print(name,json.dumps(score),flush=True)
        model.to('cpu');del model;gc.collect()
        if device=='mps':torch.mps.empty_cache()
    deployed=OnnxExtractor(args.onnx)
    p=np.concatenate([deployed.predict_proba(test.sentence.iloc[i:i+32].tolist()) for i in range(0,len(test),32)])
    predictions['deployed_int8']=p.argmax(axis=1);metrics['deployed_int8']=relation_scores(gold,predictions['deployed_int8'])
    np.save(args.output/'deployed_int8_probabilities.npy',p)
    primary=predictions['fine_tuned_weighted_seed42']
    rng=np.random.default_rng(protocol['bootstrap_seed']);groups=test.group.unique()
    indices={g:np.flatnonzero(test.group.to_numpy()==g) for g in groups}
    boot=[]
    for _ in range(protocol['bootstrap_resamples']):
        ix=np.concatenate([indices[g] for g in rng.choice(groups,len(groups),replace=True)])
        a=edge_f1(gold[ix],primary[ix]);b=edge_f1(gold[ix],ml[ix]);c=edge_f1(gold[ix],predictions['deployed_int8'][ix])
        boot.append([a,a-b,c])
    boot=np.asarray(boot)
    case=graph_case(deployed)
    probe=test.sentence.iloc[:20].tolist();deployed.predict_proba(probe[:1])
    timings=[]
    for sentence in probe:
        start=time.perf_counter();deployed.predict_proba([sentence]);timings.append(time.perf_counter()-start)
    seed_scores=[metrics[f'fine_tuned_weighted_seed{s}']['edge_f1'] for s in [42,17,2026]]
    result={'status':'HELD_OUT_EVALUATED','protocol':protocol,'test_rows':len(test),'test_groups':len(groups),
        'test_sha256':file_sha(path),'reference_partitions_sha256':audit['frozen_reference_partition_sha256'],
        'test':metrics,'selected_checkpoints':selected,
        'primary_edge_f1_bootstrap_95ci':np.quantile(boot[:,0],[.025,.975]).tolist(),
        'primary_minus_ml_edge_f1':metrics['fine_tuned_weighted_seed42']['edge_f1']-metrics['tfidf_lr']['edge_f1'],
        'paired_delta_bootstrap_95ci':np.quantile(boot[:,1],[.025,.975]).tolist(),
        'deployed_edge_f1_bootstrap_95ci':np.quantile(boot[:,2],[.025,.975]).tolist(),
        'seed_stability':{'seeds':[42,17,2026],'edge_f1':seed_scores,'mean':float(np.mean(seed_scores)),'sample_sd':float(np.std(seed_scores,ddof=1))},
        'quantized_disagreement_count':int(np.sum(primary!=predictions['deployed_int8'])),
        'deployment':{'model_bytes':(args.onnx/'model.int8.onnx').stat().st_size,'median_single_sentence_ms':float(np.median(timings)*1000),
            'latency_scope':'Local CPU, two ONNX threads, 20 sequential warm single-sentence probes; not cloud benchmark'},
        'controlled_graph_case':case,
        'uncertainty_scope':'Sentence-group bootstrap conditional on fitted model; no document IDs. Seed SD is separate.',
        'limitations':['Given entity spans','Derived three-class task, not official 19-class SemEval metric',
            'No real-domain causal identification','Scores are uncalibrated','Controlled graph case does not establish external graph accuracy']}
    saved=test[['group','label']].copy()
    for name,pred in predictions.items():saved[name]=pred
    saved.to_csv(args.output/'predictions.csv',index=False)
    (args.output/'metrics.json').write_text(json.dumps(result,indent=2))
    (args.output/'manifest.json').write_text(json.dumps({p.name:file_sha(p) for p in args.output.iterdir() if p.is_file()},indent=2))
    print(json.dumps({'status':'PASS','primary':metrics['fine_tuned_weighted_seed42'],'deployed':metrics['deployed_int8'],
        'seed_stability':result['seed_stability'],'paired_ci':result['paired_delta_bootstrap_95ci']},indent=2),flush=True)


if __name__=='__main__':main()
