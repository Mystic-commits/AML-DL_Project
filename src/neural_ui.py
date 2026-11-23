"""Streamlit views for neural inference and the reviewed neural/symbolic walkthrough."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st

from src.causal import build_graph, observations, oracle, query
from src.hybrid import candidate_record
from src.reviewed_network import fit_reviewed_network

ROOT = Path(__file__).resolve().parents[1]


@st.cache_resource(show_spinner='Loading the verified neural model…')
def _load_neural_model(archive_sha256):
    from src.neural_runtime import OnnxExtractor, ensure_bundle
    manifest = json.loads((ROOT/'artifacts/neural/release.json').read_text())
    if manifest['archive_sha256'] != archive_sha256:
        raise ValueError('Release manifest changed while loading')
    return OnnxExtractor(ensure_bundle(manifest))


def get_neural_model():
    manifest = json.loads((ROOT/'artifacts/neural/release.json').read_text())
    return _load_neural_model(manifest['archive_sha256'])


def render_neural_lab():
    st.title('Language, then structure.')
    st.write('DistilBERT reads a sentence and suggests the direction between two named entities. Review any suggestion before adding it to a causal graph.')
    st.caption('Fine-tuned language encoder, not a chat API. Entity spans are supplied. Scores are uncalibrated and do not establish causal truth.')
    preset = st.selectbox('Try an example', ['Cause and effect','Reverse wording','No causal statement','Write my own'])
    examples = {'Cause and effect':'<e1>Heat</e1> causes <e2>evaporation</e2>.',
                'Reverse wording':'<e1>Flooding</e1> is caused by <e2>rain</e2>.',
                'No causal statement':'The <e1>water</e1> is in the <e2>glass</e2>.'}
    sentence = st.text_area('Mark exactly two entities', examples.get(preset,''), key=f'neural_text_{preset}', height=115)
    st.caption('Put <e1>…</e1> around the first entity and <e2>…</e2> around the second, as in the benchmark. Keep one sentence per prediction.')
    doc = st.text_input('Source document ID', 'interactive_document')
    if st.button('Extract relation', type='primary'):
        try:
            probabilities = get_neural_model().predict_proba([sentence])[0]
            label = int(probabilities.argmax())
            names = ['No causal relation predicted','Entity 1 causes entity 2','Entity 2 causes entity 1']
            st.subheader(names[label])
            st.metric('Uncalibrated class score',f'{probabilities[label]:.3f}')
            st.dataframe(pd.DataFrame({'Class':['No relation','e1 → e2','e2 → e1'],'Score':probabilities}),hide_index=True)
            row = candidate_record(sentence,label,probabilities,doc,1)
            st.session_state.pop('classified_candidate',None)
            if row:
                st.session_state['classified_candidate'] = row
                st.success(f'Proposed edge: {row["cause"]} → {row["effect"]}')
                st.info('Next: open Text → DAG, include the latest candidate, then approve it if justified. Approval and matching observations are required before Bayesian fitting.')
            else:
                st.info('No edge was added. You can try another sentence.')
        except (ValueError, OSError, RuntimeError) as error:
            st.error(f'Could not extract this relation: {error}')


def render_hybrid_walkthrough():
    st.title('How the hybrid works')
    st.write('Four steps: extract with DistilBERT, review the graph, learn probabilities from data, and compare watching with taking action.')
    st.info('This is a controlled, made-up rain/sprinkler example. The known generating rules let us check calculations. Real-world causal conclusions need independent evidence.')
    st.subheader('1. Let the neural model read the sentences')
    sentences = ['<e1>Rain</e1> causes <e2>sprinkler</e2>.',
                 '<e1>Rain</e1> causes <e2>wet grass</e2>.',
                 '<e1>Sprinkler</e1> causes <e2>wet grass</e2>.',
                 '<e1>Wet grass</e1> causes <e2>rain</e2>.']
    st.code('\n'.join(s.replace('<e1>','').replace('</e1>','').replace('<e2>','').replace('</e2>','') for s in sentences),language=None)
    st.caption('Rain affects sprinkler use: in this generator, it makes sprinkler use less likely. The last sentence is a deliberately conflicting claim. The neural model reads the language; it does not know that the last causal claim is wrong.')
    if st.button('Extract with DistilBERT',type='primary',key='hybrid_extract'):
        try:
            probs = get_neural_model().predict_proba(sentences)
            rows = [candidate_record(t,int(p.argmax()),p,'synthetic_walkthrough',i+1) for i,(t,p) in enumerate(zip(sentences,probs))]
            st.session_state['hybrid_candidates'] = [r for r in rows if r is not None]
            st.session_state.pop('hybrid_fit',None)
        except (ValueError, OSError, RuntimeError) as error:
            st.error(str(error))
    rows = st.session_state.get('hybrid_candidates')
    if rows is None:
        st.caption('Press Extract with DistilBERT to begin. Nothing enters the graph yet.')
        return
    if not rows:
        st.info('The model proposed no edges for this input.'); return
    st.dataframe(pd.DataFrame(rows)[['cause','effect','uncalibrated_score','sentence_id']],hide_index=True)
    st.subheader('2. Review and check the graph')
    reviewed = st.checkbox('I have read these suggestions. Check their graph structure.',key='hybrid_review')
    if not reviewed:
        st.caption('Tick the review box to continue. A high neural score is not automatic approval.'); return
    graph, accepted, rejected = build_graph(rows)
    left,right = st.columns([1.4,1])
    with left:
        fig,ax=plt.subplots(figsize=(7,3),facecolor='#F8F4EC')
        pos={'rain':(0,1),'sprinkler':(1,1),'wet_grass':(.5,0)}
        nx.draw_networkx(graph,pos,ax=ax,node_color='#BF092F',font_color='white',edge_color='#594832',node_size=2600,arrowsize=22,font_size=10)
        ax.margins(.25);ax.axis('off');st.pyplot(fig);plt.close(fig)
    with right:
        st.metric('Relationships kept',len(accepted));st.metric('Relationships rejected',len(rejected))
        if rejected: st.dataframe(pd.DataFrame(rejected)[['cause','effect','reason']],hide_index=True)
    st.caption('Wet grass → rain would close a loop. This static Bayesian network requires a graph without loops. The check follows sentence order; it does not prove causal truth.')
    if set(graph.edges()) != {('rain','sprinkler'),('rain','wet_grass'),('sprinkler','wet_grass')}:
        st.warning('The predicted structure differs from the teaching generator. Use Text → DAG to review it individually; we cannot fit the known teaching data as if the correct graph had been recovered.');return
    st.subheader('3. Learn probabilities from example data')
    st.write('Use 6,000 made-up records of rain, sprinkler use and wet grass. The Bayesian network learns probability tables from these records, not from neural confidence scores.')
    assumptions = st.checkbox('Use the known teaching graph, include rain as a shared cause, and assume the other mechanisms stay stable.',key='hybrid_assumptions')
    if st.button('Learn from the example data',disabled=not assumptions,key='hybrid_fit_button'):
        network,audit=fit_reviewed_network([dict(r,approved=True) for r in accepted],observations(),dict(
            reviewed_by='Interactive reviewer',causal_rationale='Supplied synthetic generator; rain suppresses sprinkler use and both variables affect wet grass.',observed_confounders=True,stable_mechanisms=True))
        st.session_state['hybrid_fit']=(network,audit)
    if not assumptions or 'hybrid_fit' not in st.session_state:
        st.caption('Confirm the teaching assumptions and press Learn from the example data.');return
    network,audit=st.session_state['hybrid_fit']
    st.success('Ready. The neural predictions have passed review and structure checks, and the Bayesian tables are fitted.')
    st.subheader('4. Compare watching with taking action')
    value=st.radio('Choose a sprinkler setting',[0,1],format_func=lambda x:'OFF (0)' if x==0 else 'ON (1)',horizontal=True,key='hybrid_state')
    observed=query(variable='sprinkler',value=value,network=network)
    intervened=query(variable='sprinkler',value=value,intervention=True,network=network)
    a,b,c=st.columns(3);a.metric('Observed',f'{observed:.3f}');b.metric('Forced (do)',f'{intervened:.3f}');c.metric('Known answer',f'{oracle(value):.3f}')
    setting='on' if value else 'off'
    st.write(f'**Watching:** among records where the sprinkler happened to be {setting}, how often was the grass wet? Rain influences both variables.')
    st.write(f'**Taking action:** force the sprinkler {setting}. Rain no longer chooses its setting, but rain can still wet the grass. The model estimates a {intervened*100:.1f}% chance of wet grass.')
    st.caption('Small differences from the known answer reflect finite sampling and Bayesian smoothing. This is a population intervention query, not an individual counterfactual.')
    with st.expander('Show learned probability tables and audit'):
        for cpd in network.get_cpds(): st.code(str(cpd),language=None)
        st.download_button('Download reviewed network audit',json.dumps(audit,indent=2),'hybrid_network_audit.json','application/json')


def render_final_results():
    """Read immutable experiment evidence; never trigger training from the interface."""
    report = json.loads((ROOT/'artifacts/phase2/final_evaluation/metrics.json').read_text())
    st.subheader('Phase 2: held-out directed extraction')
    names = {'lexical':'Lexical rules', 'tfidf_lr':'TF-IDF + Logistic Regression',
        'fine_tuned_weighted_seed42':'DistilBERT weighted — primary',
        'frozen_encoder_weighted_seed42':'Frozen encoder + trained head',
        'fine_tuned_unweighted_seed42':'DistilBERT unweighted — ablation',
        'deployed_int8':'Hosted 8-bit model'}
    rows = [{'Model': label, **{k:report['test'][key][k] for k in ['edge_precision','edge_recall','edge_f1','macro_f1']}} for key,label in names.items()]
    st.dataframe(pd.DataFrame(rows).set_index('Model').style.format('{:.3f}'), use_container_width=True)
    low,high = report['primary_edge_f1_bootstrap_95ci']
    stability = report['seed_stability']
    st.caption(f"Same 2,700 held-out pairs and supplied entity spans. Primary seed 42 was fixed before testing; development F1 chose its epoch. Primary edge-F1 95% sentence-group bootstrap interval: {low:.3f}–{high:.3f} (1,000 resamples).")
    st.info(f"Class weighting did not help in this seed-42 comparison: the unweighted ablation scored 0.899. It is reported separately rather than selected using test scores. Three weighted seeds: mean {stability['mean']:.3f}, sample SD {stability['sample_sd']:.4f}. The frozen control was not separately tuned.")
    with st.expander('Training curves, replication seeds and audit'):
        st.image(str(ROOT/'artifacts/phase2/development/final_20261123/learning_curves.png'))
        seed_rows = [{'Seed': seed, 'Test edge F1':report['test'][f'fine_tuned_weighted_seed{seed}']['edge_f1']} for seed in [42,17,2026]]
        st.dataframe(pd.DataFrame(seed_rows), hide_index=True)
        st.write('The graph walkthrough is a single controlled synthetic case, not a held-out real-world causal graph benchmark. Neural confidence is uncalibrated. Bayesian tables use observations, not classifier scores.')
        st.download_button('Download final evaluation evidence',json.dumps(report,indent=2),'phase2_final_metrics.json','application/json')
